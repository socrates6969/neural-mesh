//! F3 RESOURCE-REDUCTION: per-decision cost of the router/mesh and the smallest viable brain.
//!
//! Part 1  Pareto: accuracy vs bytes vs ns vs allocations for keyword / warm(hebr) / guarded(hebr, the
//!         recommended default) / coupled / +knowledge, plus the compact guarded brain proposed here.
//! Part 2  Representation search: interned ids, f32/u16/u8 cells, guard-count precision, hashed ids,
//!         row budgets (drop cold features). Tuned on seeds 1-8, evaluated on held-out 31-60 and frozen
//!         on 61-90: paired 95% t-CI vs the f64 crate router, TOST equivalence (margin 0.003), Holm.
//! Part 3  Agents per GiB: K brains actually built, commit charge measured, shared vocab / hashed ids.
//!
//! CPU only, std only, no GPU, no new crates. The only place a GPU could ever matter is an OPTIONAL
//! embedding model (knowledge layer); the router and mesh are a few hundred ns..3 us of scalar CPU work.
//!
//! Usage: cargo run --release --bin f3_brain [-- --out results/f3_brain.json] [--quick]

#[path = "f3_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use common::alloc_count;
use common::brain::*;
use common::*;
use neural_mesh::baselines::KeywordRouter;
use neural_mesh::coupling::{CoupledBrain, CouplingConfig, heb_robust_cfg};
use neural_mesh::defaults::{recommended_guard, recommended_inner, recommended_router};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::knowledge::{AnchorRouter, FeatMode, Knowledge, KnowledgeConfig, KnowledgeRouter, hash_embed};
use neural_mesh::prior::{Evidence, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::Scenario;
use neural_mesh::stats::holm_adjust;
use serde_json::{Value, json};

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

const DIM: usize = 32;
const GIB: f64 = (1u64 << 30) as f64;

// =============================================================================================
// policies over an Env

trait Pol {
    fn pick(&mut self, env: &Env, t: usize, c: &[usize], rng: &mut Rng) -> usize;
    fn fb(&mut self, env: &Env, t: usize, a: usize, r: f64);
}

struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, env: &Env, t: usize, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(env.task(t), c, rng)
    }
    fn fb(&mut self, env: &Env, t: usize, a: usize, r: f64) {
        self.0.update(env.task(t), a, r)
    }
}

struct MeshPol {
    m: NeuralMesh<HebbianRouter>,
    team: Vec<usize>,
    k: usize,
}
impl Pol for MeshPol {
    fn pick(&mut self, env: &Env, t: usize, c: &[usize], rng: &mut Rng) -> usize {
        self.team = self.m.route_k(env.task(t), c, rng, self.k);
        self.team[0]
    }
    fn fb(&mut self, env: &Env, t: usize, _a: usize, r: f64) {
        let team = std::mem::take(&mut self.team);
        self.m.feedback(env.task(t), &team, r);
    }
}

struct KnPol<R: Router>(KnowledgeRouter<R>);
impl<R: Router> Pol for KnPol<R> {
    fn pick(&mut self, env: &Env, t: usize, c: &[usize], rng: &mut Rng) -> usize {
        let task = env.task(t);
        let emb = hash_embed(task, DIM);
        self.0.set_context(&emb);
        self.0.select(task, c, rng)
    }
    fn fb(&mut self, env: &Env, t: usize, a: usize, r: f64) {
        self.0.update(env.task(t), a, r)
    }
}

/// Compact brain driven through pre-interned ids (ids API: no string hashing per decision).
struct IdsPol {
    b: Box<dyn Brain>,
    ids: Arc<Vec<IdBuf>>,
}
impl Pol for IdsPol {
    fn pick(&mut self, _env: &Env, t: usize, c: &[usize], rng: &mut Rng) -> usize {
        self.b.route(self.ids[t].as_slice(), c, rng)
    }
    fn fb(&mut self, _env: &Env, t: usize, a: usize, r: f64) {
        self.b.feedback(self.ids[t].as_slice(), a, r)
    }
}

/// Precomputed outcome tables so timing excludes the world.
struct Tables {
    n: usize,
    len: usize,
    prob: Vec<f64>,
    out: Vec<u8>,
}
impl Tables {
    fn new(env: &Env) -> Tables {
        let (n, len) = (env.n_agents(), env.len());
        let mut prob = vec![0.0; n * len];
        let mut out = vec![0u8; n * len];
        for t in 0..len {
            for a in 0..n {
                prob[t * n + a] = env.prob(t, a);
                out[t * n + a] = env.outcome(t, a) as u8;
            }
        }
        Tables { n, len, prob, out }
    }
}

fn drive(pol: &mut dyn Pol, env: &Env, tb: &Tables, cands: &[usize], rng: &mut Rng, range: std::ops::Range<usize>, cum: &mut f64) {
    for t in range {
        let a = pol.pick(env, t, cands, rng);
        *cum += tb.prob[t * tb.n + a];
        pol.fb(env, t, a, tb.out[t * tb.n + a] as f64);
    }
}

fn run_tab(pol: &mut dyn Pol, env: &Env, tb: &Tables, seed: u64) -> RunStats {
    let cands: Vec<usize> = (0..tb.n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = RunStats::default();
    drive(pol, env, tb, &cands, &mut rng, 0..250, &mut cum);
    out.h250 = cum / 250.0;
    drive(pol, env, tb, &cands, &mut rng, 250..500, &mut cum);
    out.h500 = cum / 500.0;
    drive(pol, env, tb, &cands, &mut rng, 500..tb.len, &mut cum);
    out.full = cum / tb.len as f64;
    out
}

// =============================================================================================
// cost measurement (single-threaded, allocator-counted memory pass + wall-clock pass)

#[derive(Clone, Debug, Default)]
struct Cost {
    live: i64,
    blocks: i64,
    peak: i64,
    allocs_per_dec_all: f64,
    allocs_per_dec_steady: f64,
    ns_med: f64,
    ns_min: f64,
    ns_first500: f64,
    ns_last500: f64,
}

type Make<'a> = &'a dyn Fn(&Env) -> Box<dyn Pol>;

fn cost_of(make: Make, env: &Env, tb: &Tables, seed: u64, reps: usize) -> Cost {
    let cands: Vec<usize> = (0..tb.n).collect();
    let len = tb.len;
    let mut c = Cost::default();
    // ---- memory + allocation pass
    {
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let mut cum = 0.0;
        alloc_count::start();
        let mut pol = make(env);
        drive(pol.as_mut(), env, tb, &cands, &mut rng, 0..len / 2, &mut cum);
        let s1 = alloc_count::snap();
        drive(pol.as_mut(), env, tb, &cands, &mut rng, len / 2..len, &mut cum);
        let s2 = alloc_count::stop();
        c.live = s2.live;
        c.blocks = s2.blocks;
        c.peak = s2.peak;
        c.allocs_per_dec_all = (s2.allocs + s2.reallocs) as f64 / len as f64;
        c.allocs_per_dec_steady = ((s2.allocs + s2.reallocs) - (s1.allocs + s1.reallocs)) as f64 / (len - len / 2) as f64;
        drop(pol);
    }
    if reps == 0 {
        return c;
    }
    // ---- timing pass (allocator counting off)
    let mut tot = vec![];
    let mut first = vec![];
    let mut last = vec![];
    for rep in 0..reps + 1 {
        let mut pol = make(env);
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let mut cum = 0.0;
        let t0 = Instant::now();
        drive(pol.as_mut(), env, tb, &cands, &mut rng, 0..500, &mut cum);
        let t1 = Instant::now();
        drive(pol.as_mut(), env, tb, &cands, &mut rng, 500..len - 500, &mut cum);
        let t2 = Instant::now();
        drive(pol.as_mut(), env, tb, &cands, &mut rng, len - 500..len, &mut cum);
        let t3 = Instant::now();
        std::hint::black_box(cum);
        if rep == 0 {
            continue; // warm-up
        }
        tot.push((t3 - t0).as_nanos() as f64 / len as f64);
        first.push((t1 - t0).as_nanos() as f64 / 500.0);
        last.push((t3 - t2).as_nanos() as f64 / 500.0);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    c.ns_min = tot.iter().cloned().fold(f64::INFINITY, f64::min);
    c.ns_med = med(&mut tot);
    c.ns_first500 = med(&mut first);
    c.ns_last500 = med(&mut last);
    c
}

// =============================================================================================
// world helpers

fn world_features(env: &Env) -> Vec<String> {
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut out = vec![];
    for t in 0..env.len() {
        for f in &env.task(t).features {
            if seen.insert(f.as_str(), ()).is_none() {
                out.push(f.clone());
            }
        }
    }
    out
}

/// Shared vocab over the world's features + keyword map ids + per-task ids.
struct Pre {
    vocab: Vocab,
    kw: Arc<[(u32, u16)]>,
    ids: Arc<Vec<IdBuf>>,
    distinct: usize,
}
fn pre(env: &Env, mode: IdMode) -> Pre {
    let mut vocab = Vocab::new();
    let mut ids = Vec::with_capacity(env.len());
    for t in 0..env.len() {
        let task = env.task(t);
        let mut b = IdBuf { a: [0; MAXF], len: 0 };
        for f in task.features.iter().take(MAXF) {
            b.a[b.len] = if mode == IdMode::Table { vocab.intern(f) } else { hash32(f) };
            b.len += 1;
        }
        ids.push(b);
    }
    let distinct = world_features(env).len();
    let mut v: Vec<(u32, u16)> = env
        .kw_map()
        .iter()
        .map(|(k, &a)| (if mode == IdMode::Table { vocab.intern(k) } else { hash32(k) }, a as u16))
        .collect();
    v.sort_by_key(|e| e.0);
    Pre { vocab, kw: v.into(), ids: Arc::new(ids), distinct }
}

// =============================================================================================
// Pareto configurations (crate objects)

fn coupled_cfg() -> CouplingConfig {
    CouplingConfig {
        hebbian: recommended_inner(),
        guard: Some(recommended_guard()),
        health: true,
        health_alpha: 0.2,
        health_threshold: 0.3,
        retry_every: 20,
        shield: true,
        probe_gate: true,
        ..CouplingConfig::default()
    }
}

fn mk_keyword(env: &Env) -> Box<dyn Pol> {
    Box::new(Plain(KeywordRouter::new(env.kw_map(), env.n_agents())))
}
fn mk_warm(env: &Env) -> Box<dyn Pol> {
    let n = env.n_agents();
    Box::new(Plain(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), env.kw_map(), n, 1.0).with_evidence(Evidence::MeanFeature)))
}
fn mk_guarded(env: &Env) -> Box<dyn Pol> {
    Box::new(Plain(recommended_router(env.kw_map(), env.n_agents())))
}
fn mk_coupled(env: &Env) -> Box<dyn Pol> {
    Box::new(Plain(CoupledBrain::new(env.n_agents(), env.kw_map(), coupled_cfg())))
}
fn mk_mesh1(env: &Env) -> Box<dyn Pol> {
    Box::new(MeshPol { m: NeuralMesh::new(env.n_agents(), HebbianRouter::new(env.n_agents(), heb_robust_cfg()), MeshConfig::default()), team: vec![], k: 1 })
}
fn mk_mesh3(env: &Env) -> Box<dyn Pol> {
    let cfg = MeshConfig { team_size: 3, ..MeshConfig::default() };
    Box::new(MeshPol { m: NeuralMesh::new(env.n_agents(), HebbianRouter::new(env.n_agents(), heb_robust_cfg()), cfg), team: vec![], k: 3 })
}
fn mk_knowledge(env: &Env) -> Box<dyn Pol> {
    let n = env.n_agents();
    let mut kc = KnowledgeConfig::new(DIM);
    kc.k = 8;
    kc.sim_min = 0.6;
    kc.feat = FeatMode::Hidden;
    kc.vote = None;
    let kn = Knowledge::new_shared(kc);
    let anchors = std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()));
    let inner = AnchorRouter::new(HebbianRouter::new(n, recommended_inner()), anchors.clone());
    let front = Guarded::new(inner, env.kw_map(), n, recommended_guard());
    Box::new(KnPol(KnowledgeRouter::new(front, kn, anchors)))
}

// compact schemes -------------------------------------------------------------------------------

#[derive(Clone)]
struct Scheme {
    name: String,
    mode: IdMode,
    maker: Maker,
    rows: Rows,
    evict: Evict,
    /// frequency-gated admission (sightings before a feature gets a row; <= 1 = off)
    admit: u8,
}

#[derive(Clone, Copy)]
enum Rows {
    All,
    /// fraction of the distinct features of the run
    Frac(f64),
    Abs(usize),
}

fn sch(name: &str, mode: IdMode, maker: Maker) -> Scheme {
    Scheme { name: name.into(), mode, maker, rows: Rows::All, evict: Evict::None, admit: 0 }
}
impl Scheme {
    fn lim(mut self, rows: Rows, evict: Evict) -> Self {
        self.rows = rows;
        self.evict = evict;
        self
    }
    fn adm(mut self, a: u8) -> Self {
        self.admit = a;
        self
    }
}

type U16 = Fx16;
fn q_sparse(s: u32) -> Maker {
    match s {
        64 => mks::<U16, Q16<64>>,
        256 => mks::<U16, Q16<256>>,
        1024 => mks::<U16, Q16<1024>>,
        _ => mks::<U16, Q16<4096>>,
    }
}
fn q_maker(s: u32) -> Maker {
    match s {
        64 => mk::<U16, Q16<64>>,
        256 => mk::<U16, Q16<256>>,
        1024 => mk::<U16, Q16<1024>>,
        _ => mk::<U16, Q16<4096>>,
    }
}

impl Scheme {
    fn rows_budget(&self, distinct: usize) -> Budget {
        let m = match self.rows {
            Rows::All => 0,
            Rows::Frac(f) => ((distinct as f64 * f).ceil() as usize).max(2),
            Rows::Abs(a) => a,
        };
        if m == 0 && self.admit <= 1 { None } else { Some((m, self.evict, self.admit)) }
    }
    /// Router (string API) for accuracy runs.
    fn router(&self, env: &Env, distinct: usize) -> Box<dyn Pol> {
        Box::new(Plain(DynRouter::new(env.n_agents(), &env.kw_map(), self.mode, self.maker, self.rows_budget(distinct), &self.name)))
    }
}

// =============================================================================================
// self tests

fn selftest() {
    // 1. compact f64/f64 == crate Guarded<HebbianRouter> decision for decision, incl. rng use
    for (seed, spec) in [(3u64, WorldSpec::Sim(Scenario::Fixed)), (4, WorldSpec::Sim(Scenario::Random)), (5, WorldSpec::Big), (6, WorldSpec::BigTail)] {
        let env = spec.make(seed);
        let n = env.n_agents();
        let mut a = recommended_router(env.kw_map(), n);
        let mut b = DynRouter::new(n, &env.kw_map(), IdMode::Table, mk::<f64, f64>, None, "c64");
        let cands: Vec<usize> = (0..n).collect();
        let (mut r1, mut r2) = (Rng::new(7), Rng::new(7));
        let mut kinds = [0usize; 3];
        for t in 0..env.len() {
            let task = env.task(t);
            let x = a.select(task, &cands, &mut r1);
            let y = b.select(task, &cands, &mut r2);
            assert_eq!(x, y, "compact f64 diverged from crate Guarded at t={t} in {}", spec.label());
            let rw = env.outcome(t, x);
            a.update(task, x, rw);
            b.update(task, y, rw);
            kinds[0] += 1;
        }
        assert_eq!(r1.next_u64(), r2.next_u64(), "rng streams diverged");
        println!("[selftest] compact f64/f64 == crate guarded(hebr) on {} ({} decisions, identical rng stream)", spec.label(), kinds[0]);
    }
    // 1b. sparse rows == dense rows (bit-identical decisions) for f64 and for the quantised cells
    for (seed, spec) in [(13u64, WorldSpec::Sim(Scenario::Fixed)), (14, WorldSpec::Big), (15, WorldSpec::BigTail)] {
        let env = spec.make(seed);
        let n = env.n_agents();
        let cands: Vec<usize> = (0..n).collect();
        for (lbl, md, ms) in [("f64/f64", mk::<f64, f64> as Maker, mks::<f64, f64> as Maker), ("u16/q16", mk::<U16, Q16<1024>> as Maker, mks::<U16, Q16<1024>> as Maker)] {
            let mut a = DynRouter::new(n, &env.kw_map(), IdMode::Table, md, None, "dense");
            let mut b = DynRouter::new(n, &env.kw_map(), IdMode::Table, ms, None, "sparse");
            let (mut r1, mut r2) = (Rng::new(5), Rng::new(5));
            for t in 0..env.len() {
                let task = env.task(t);
                let x = a.select(task, &cands, &mut r1);
                let y = b.select(task, &cands, &mut r2);
                assert_eq!(x, y, "sparse diverged from dense ({lbl}) at t={t} in {}", spec.label());
                let rw = env.outcome(t, x);
                a.update(task, x, rw);
                b.update(task, y, rw);
            }
        }
        println!("[selftest] sparse rows == dense rows (f64/f64 and u16/q16) on {}", spec.label());
    }
    // 2. hashed ids give the same decisions as the table unless two features collide
    let env = WorldSpec::Big.make(9);
    let n = env.n_agents();
    let mut a = DynRouter::new(n, &env.kw_map(), IdMode::Table, mk::<f64, f64>, None, "t");
    let mut b = DynRouter::new(n, &env.kw_map(), IdMode::Hash32, mk::<f64, f64>, None, "h");
    let cands: Vec<usize> = (0..n).collect();
    let (mut r1, mut r2) = (Rng::new(1), Rng::new(1));
    for t in 0..env.len() {
        let task = env.task(t);
        let x = a.select(task, &cands, &mut r1);
        let y = b.select(task, &cands, &mut r2);
        assert_eq!(x, y);
        let rw = env.outcome(t, x);
        a.update(task, x, rw);
        b.update(task, y, rw);
    }
    println!("[selftest] hash32 ids == interned ids on big-32agents (no collisions among {} features)", world_features(&env).len());
    // 2b. software binary16 round-trips every finite half exactly; bf16 keeps the top 16 bits
    for h in 0u16..=0xffff {
        if (h >> 10) & 0x1f == 0x1f {
            continue; // inf/nan
        }
        assert_eq!(f32_to_f16(f16_to_f32(h)), h, "f16 roundtrip {h:#x}");
    }
    assert!((F16::enc(0.1).dec() - 0.1).abs() < 5e-5 && (Bf16::enc(0.1).dec() - 0.1).abs() < 4e-4);
    println!("[selftest] binary16 conversion round-trips all finite values; bf16 ok");
    // 3. counting allocator sanity
    let (v, s) = alloc_count::measure(|| vec![0u8; 100_000]);
    assert!(s.live >= 100_000 && s.blocks >= 1);
    drop(v);
    let (_, s) = alloc_count::measure(|| {
        let v = vec![1u32; 10];
        std::hint::black_box(&v);
    });
    assert_eq!(s.live, 0);
    assert_eq!(s.allocs, 1);
    println!("[selftest] counting allocator ok");
    // 4. eviction keeps invariants (budget respected, still routes)
    let env = WorldSpec::BigTail.make(11);
    let n = env.n_agents();
    let mut c = DynRouter::new(n, &env.kw_map(), IdMode::Table, mk::<U16, Q16<1024>>, Some((200, Evict::Lru, 0)), "lru");
    let (mut r, cands) = (Rng::new(3), (0..n).collect::<Vec<_>>());
    for t in 0..env.len() {
        let a = c.select(env.task(t), &cands, &mut r);
        c.update(env.task(t), a, env.outcome(t, a));
    }
    assert!(c.b.rows() <= 200 && c.b.evictions() > 100);
    println!("[selftest] row budget 200 respected on the long-tail world ({} evictions)", c.b.evictions());
}

/// Per-block allocator overhead of the platform heap: allocate many small blocks, read commit charge.
fn calibrate_block_overhead() -> (f64, f64) {
    let sizes = [24usize, 64, 200];
    let mut res = vec![];
    for &sz in &sizes {
        let (c0, _) = proc_mem();
        let n = 300_000;
        let mut v: Vec<Box<[u8]>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut b = vec![0u8; sz].into_boxed_slice();
            b[0] = i as u8;
            v.push(b);
        }
        let (c1, _) = proc_mem();
        let per = (c1 as f64 - c0 as f64) / n as f64 - (std::mem::size_of::<Box<[u8]>>() as f64);
        res.push((sz, per));
        drop(v);
    }
    // overhead = per-block commit - requested size
    let ov: Vec<f64> = res.iter().map(|(s, p)| p - *s as f64).collect();
    println!("[calibrate] windows heap commit per block for sizes {:?}: {:?} => overhead {:?} B", sizes, res.iter().map(|x| x.1.round()).collect::<Vec<_>>(), ov.iter().map(|x| x.round()).collect::<Vec<_>>());
    let mean = ov.iter().sum::<f64>() / ov.len() as f64;
    (mean, ov.iter().cloned().fold(f64::INFINITY, f64::min))
}

// =============================================================================================
// stats helpers

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, v.sqrt())
}

struct Verdict {
    label: &'static str,
}
fn verdict(c: &Cmp, holm_equiv: f64) -> Verdict {
    let inside = c.lo > -MARGIN && c.hi < MARGIN;
    let label = if inside && holm_equiv < 0.05 {
        "EQUIVALENT"
    } else if c.hi < 0.0 {
        "LOSS"
    } else if c.lo > 0.0 {
        "GAIN"
    } else {
        "inconclusive"
    };
    Verdict { label }
}

// =============================================================================================

// =============================================================================================
// PART 3 helpers: K real brains, measured in-process (allocator) and in a fresh child process (OS commit)

fn part3_worlds() -> Vec<WorldSpec> {
    vec![WorldSpec::Sim(Scenario::Fixed), WorldSpec::Big, WorldSpec::BigTail]
}

fn kind_list(s_star: u32) -> Vec<(String, Option<Scheme>)> {
    vec![
        ("crate guarded(hebr) f64 [String keys, BTreeMap]".into(), None),
        ("c f32/f32 dense".into(), Some(sch("k", IdMode::Table, mk::<f32, f32>))),
        ("c u16/q16 dense".into(), Some(sch("k", IdMode::Table, q_maker(s_star)))),
        ("c u16/q16 sparse".into(), Some(sch("k", IdMode::Table, q_sparse(s_star)))),
        ("c u16/q16 sparse hash32".into(), Some(sch("k", IdMode::Hash32, q_sparse(s_star)))),
        ("c u16/q16 sparse + admit>=2 + LRU 512".into(), Some(sch("k", IdMode::Table, q_sparse(s_star)).adm(2).lim(Rows::Abs(512), Evict::Lru))),
    ]
}

/// Child process: build K brains in a clean heap and report commit charge / working set per brain.
fn probe_main(args: &[String]) {
    // --probe <kind> <world> <private:0|1> <K or 0=auto> <s_star> <seed>
    let ki: usize = args[2].parse().unwrap();
    let wi: usize = args[3].parse().unwrap();
    let private = args[4] == "1";
    let k_req: usize = args[5].parse().unwrap();
    let s_star: u32 = args[6].parse().unwrap();
    let seed: u64 = args[7].parse().unwrap();
    let kinds = kind_list(s_star);
    let spec = part3_worlds()[wi].clone();
    let env = spec.make(seed);
    let tb = Tables::new(&env);
    let n = tb.n;
    let cands: Vec<usize> = (0..n).collect();
    let (k, c0, w0, c1, w1);
    match &kinds[ki].1 {
        None => {
            let sz_guess = if matches!(spec, WorldSpec::Sim(_)) { 17_000 } else if matches!(spec, WorldSpec::Big) { 250_000 } else { 1_900_000 };
            k = if k_req > 0 { k_req } else { (40_000_000 / sz_guess).clamp(24, 160) };
            let (a0, b0) = proc_mem();
            let mut keep: Vec<Box<dyn Pol>> = Vec::with_capacity(k);
            for i in 0..k {
                let mut p = mk_guarded(&env);
                let mut rng = Rng::new(1000 + i as u64);
                let mut cum = 0.0;
                drive(p.as_mut(), &env, &tb, &cands, &mut rng, 0..tb.len, &mut cum);
                keep.push(p);
            }
            let (a1, b1) = proc_mem();
            std::hint::black_box(&keep);
            c0 = a0;
            w0 = b0;
            c1 = a1;
            w1 = b1;
        }
        Some(scheme) => {
            let p = pre(&env, scheme.mode);
            let mut b = (scheme.maker)(n, p.kw.clone(), scheme.rows_budget(p.distinct));
            let mut rng = Rng::new(7);
            for t in 0..tb.len {
                let a = b.route(p.ids[t].as_slice(), &cands, &mut rng);
                b.feedback(p.ids[t].as_slice(), a, tb.out[t * tb.n + a] as f64);
            }
            b.shrink();
            let sz = b.heap_bytes() + 64;
            k = if k_req > 0 { k_req } else { (40_000_000 / sz).clamp(300, 60_000) };
            let (a0, b0) = proc_mem();
            let mut keep: Vec<(Box<dyn Brain>, Option<Vocab>, Option<Arc<[(u32, u16)]>>)> = Vec::with_capacity(k);
            for _ in 0..k {
                let bb = b.clone_box();
                let v = if private && scheme.mode == IdMode::Table { Some(p.vocab.clone()) } else { None };
                let kwp: Option<Arc<[(u32, u16)]>> = if private { Some(p.kw.iter().cloned().collect::<Vec<_>>().into()) } else { None };
                keep.push((bb, v, kwp));
            }
            let (a1, b1) = proc_mem();
            std::hint::black_box(&keep);
            c0 = a0;
            w0 = b0;
            c1 = a1;
            w1 = b1;
        }
    }
    println!("{}", json!({"k": k, "commit_per": (c1 as f64 - c0 as f64) / k as f64, "ws_per": (w1 as f64 - w0 as f64) / k as f64}));
}

fn part3(quick: bool, s_star: u32, seed: u64, blk_ov: f64) -> Vec<Value> {
    println!("\n=== PART 3: how many brains fit per GiB (K brains actually built; live bytes from the counting allocator, commit charge from a fresh child process per row) ===");
    let exe = std::env::current_exe().unwrap();
    let kinds = kind_list(s_star);
    let mut out = vec![];
    for (wi, spec) in part3_worlds().iter().enumerate() {
        let env = spec.make(seed);
        let tb = Tables::new(&env);
        let n = tb.n;
        let cands: Vec<usize> = (0..n).collect();
        println!("\n-- world {} ({} agents, {} tasks)", spec.label(), n, tb.len);
        println!("{:<46} {:<9} {:>5} {:>10} {:>8} {:>11} {:>11} {:>14}", "brain", "sharing", "rows", "live B", "blocks", "commit B", "shared once", "brains/GiB");
        let mut modes = vec![];
        for (ki, (name, scheme)) in kinds.iter().enumerate() {
            for private in [true, false] {
                if scheme.is_none() && !private {
                    continue; // the crate router has no sharing
                }
                if let Some(s) = scheme {
                    if s.mode == IdMode::Hash32 && private {
                        // hashed ids have no vocab; "private" would only duplicate the keyword map: still report it
                    }
                }
                // ---- in-process: live bytes + blocks per brain (allocator counted)
                let (live, blocks, rows_n, shared_once);
                match scheme {
                    None => {
                        let kc = if quick { 8 } else { 24 };
                        alloc_count::start();
                        let mut keep: Vec<Box<dyn Pol>> = Vec::with_capacity(kc);
                        for i in 0..kc {
                            let mut p = mk_guarded(&env);
                            let mut rng = Rng::new(1000 + i as u64);
                            let mut cum = 0.0;
                            drive(p.as_mut(), &env, &tb, &cands, &mut rng, 0..tb.len, &mut cum);
                            keep.push(p);
                        }
                        let s = alloc_count::stop();
                        std::hint::black_box(&keep);
                        live = s.live as f64 / kc as f64;
                        blocks = s.blocks as f64 / kc as f64;
                        rows_n = 0usize;
                        shared_once = 0.0;
                    }
                    Some(sc) => {
                        let p = pre(&env, sc.mode);
                        let mut b = (sc.maker)(n, p.kw.clone(), sc.rows_budget(p.distinct));
                        let mut rng = Rng::new(7);
                        for t in 0..tb.len {
                            let a = b.route(p.ids[t].as_slice(), &cands, &mut rng);
                            b.feedback(p.ids[t].as_slice(), a, tb.out[t * tb.n + a] as f64);
                        }
                        b.shrink();
                        rows_n = b.rows();
                        let vocab_bytes = if sc.mode == IdMode::Table { alloc_count::measure(|| p.vocab.clone()).1.live } else { 0 };
                        let kw_bytes = (p.kw.len() * 8) as i64;
                        let kc = 400;
                        alloc_count::start();
                        let mut keep: Vec<(Box<dyn Brain>, Option<Vocab>, Option<Arc<[(u32, u16)]>>)> = Vec::with_capacity(kc);
                        for _ in 0..kc {
                            let bb = b.clone_box();
                            let v = if private && sc.mode == IdMode::Table { Some(p.vocab.clone()) } else { None };
                            let kwp: Option<Arc<[(u32, u16)]>> = if private { Some(p.kw.iter().cloned().collect::<Vec<_>>().into()) } else { None };
                            keep.push((bb, v, kwp));
                        }
                        let s = alloc_count::stop();
                        std::hint::black_box(&keep);
                        live = s.live as f64 / kc as f64;
                        blocks = s.blocks as f64 / kc as f64;
                        shared_once = if private { 0.0 } else { (vocab_bytes + kw_bytes) as f64 };
                    }
                }
                // ---- child process: OS commit per brain
                let k_req = if quick { 200 } else { 0 };
                let o = std::process::Command::new(&exe)
                    .args(["--probe", &ki.to_string(), &wi.to_string(), if private { "1" } else { "0" }, &k_req.to_string(), &s_star.to_string(), &seed.to_string()])
                    .output();
                let (commit, ws, kk) = match o {
                    Ok(o) if o.status.success() => {
                        let line = String::from_utf8_lossy(&o.stdout).lines().last().unwrap_or("").to_string();
                        let v: Value = serde_json::from_str(&line).unwrap_or(json!({}));
                        (v["commit_per"].as_f64().unwrap_or(0.0), v["ws_per"].as_f64().unwrap_or(0.0), v["k"].as_u64().unwrap_or(0))
                    }
                    _ => (0.0, 0.0, 0),
                };
                let est = live + blk_ov * blocks;
                let per_gib = (GIB - shared_once) / commit.max(est).max(1.0);
                println!(
                    "{:<46} {:<9} {:>5} {:>10.0} {:>8.1} {:>11.0} {:>11.0} {:>14.0}   (est live+hdr {:.0} B, ws {:.0} B, K={})",
                    name, if private { "private" } else { "shared" }, rows_n, live, blocks, commit, shared_once, per_gib, est, ws, kk
                );
                modes.push(json!({
                    "brain": name, "sharing": if private { "private vocab+kw per brain" } else { "shared vocab+kw (Arc)" }, "rows": rows_n,
                    "live_bytes_per_brain": live, "blocks_per_brain": blocks, "est_live_plus_heap_header": est,
                    "commit_bytes_per_brain_child_process": commit, "working_set_per_brain": ws, "k_child": kk,
                    "shared_once_bytes": shared_once, "brains_per_gib": per_gib,
                    "brains_per_gib_from_commit_only": (GIB - shared_once) / commit.max(1.0), "brains_per_gib_from_live_plus_hdr": (GIB - shared_once) / est.max(1.0),
                }));
            }
        }
        out.push(json!({"world": spec.label(), "agents": n, "tasks": tb.len, "modes": modes}));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("--probe") {
        probe_main(&args);
        return;
    }
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/f3_brain.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let quiet_wait: u64 = args.iter().position(|a| a == "--quiet-wait").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(if quick { 0 } else { 90 });
    let t_start = Instant::now();
    println!("# f3_brain  cores(logical)={}  quick={quick}", n_threads());
    selftest();
    let (blk_ov_mean, _blk_ov_min) = calibrate_block_overhead();

    let worlds = all_worlds();
    let held = if quick { seeds(31, 38) } else { held_seeds() };
    let confirm = if quick { seeds(61, 68) } else { confirm_seeds() };
    let tune = tune_seeds();
    let reps = if quick { 3 } else { 31 };

    // ----------------------------------------------------------------------------------------
    // PART 1: Pareto
    // ----------------------------------------------------------------------------------------
    println!("\n=== PART 1: per-decision cost Pareto (held-out seeds {}..{}; cost measured on seed {}) ===", held[0], held[held.len() - 1], held[0]);
    let q_best = q_maker(1024);
    let mut part1 = vec![];
    for spec in &worlds {
        println!("\n-- world {}", spec.label());
        let env0 = spec.make(held[0]);
        let tb0 = Tables::new(&env0);
        let q0 = wait_quiet(quiet_wait, 45.0);
        println!("   (background cpu before this world: {q0:.0}%)");
        let pre0 = pre(&env0, IdMode::Table);
        let distinct = pre0.distinct;
        let compact_u16 = sch("compact guarded u16/q16", IdMode::Table, q_best);
        let ids_arc = pre0.ids.clone();
        let kw_arc = pre0.kw.clone();
        type MkBox = Box<dyn Fn(&Env) -> Box<dyn Pol> + Sync + Send>;
        let kw_for_ids = kw_arc.clone();
        let cfgs: Vec<(String, MkBox, bool)> = vec![
            ("keyword".into(), Box::new(mk_keyword), false),
            ("warm(hebr)".into(), Box::new(mk_warm), false),
            ("guarded(hebr) [recommended default]".into(), Box::new(mk_guarded), false),
            ("coupled(guarded+health+shield+probe)".into(), Box::new(mk_coupled), false),
            ("guarded + knowledge(kn-feat hidden)".into(), Box::new(mk_knowledge), false),
            ("mesh-robust team 1 [= CLI/serve state]".into(), Box::new(mk_mesh1), false),
            ("mesh-robust team 3 (edges+health)".into(), Box::new(mk_mesh3), false),
            ("compact guarded u16/q16 (string API)".into(), { let s = compact_u16.clone(); Box::new(move |e: &Env| s.router(e, distinct)) }, true),
            ("compact guarded u16/q16 SPARSE (string API)".into(), { let s = sch("sp", IdMode::Table, q_sparse(1024)); Box::new(move |e: &Env| s.router(e, distinct)) }, true),
            ("compact guarded u16/q16 (ids API)".into(), {
                let (ia, ka) = (ids_arc.clone(), kw_for_ids.clone());
                Box::new(move |e: &Env| Box::new(IdsPol { b: q_best(e.n_agents(), ka.clone(), None), ids: ia.clone() }) as Box<dyn Pol>)
            }, true),
        ];
        // accuracy over held-out seeds (parallel, no allocator counting)
        let acc: Vec<Vec<RunStats>> = cfgs
            .iter()
            .map(|(_, mk, _)| {
                par_map(&held, |&s| {
                    let env = spec.make(s);
                    let tb = Tables::new(&env);
                    // the ids-API variant needs per-seed ids: build lazily through the string router instead
                    let mut pol = mk(&env);
                    run_tab(pol.as_mut(), &env, &tb, s)
                })
            })
            .collect();
        // NB: for the ids-API compact policy accuracy is computed with seed-0 ids (same world structure only
        // for sim worlds); to be safe we override it with the string-API accuracy (identical decisions).
        let ref_idx = 2;
        let str_idx = cfgs.iter().position(|c| c.0.starts_with("compact guarded u16/q16 (string API)")).unwrap();
        let mut rows = vec![];
        for (i, (name, mk, _)) in cfgs.iter().enumerate() {
            let acc_i = if name.contains("(ids API)") { &acc[str_idx] } else { &acc[i] };
            let full: Vec<f64> = acc_i.iter().map(|r| r.full).collect();
            let h250: Vec<f64> = acc_i.iter().map(|r| r.h250).collect();
            let refv: Vec<f64> = acc[ref_idx].iter().map(|r| r.full).collect();
            let (m, lo, hi) = ci_half(&full);
            let c = compare(&full, &refv);
            let bg = cpu_busy_pct(120);
            let cost = cost_of(&**mk, &env0, &tb0, held[0], reps);
            let overhead = blk_ov_mean.max(0.0);
            let bytes_rss_est = cost.live as f64 + overhead * cost.blocks as f64;
            println!(
                "{:<44} acc {:.4} [{:.4},{:.4}] vs-guarded {:+.4}  state {:>9.0} B live ({:>5} blocks, ~{:>9.0} B w/ heap hdr)  peak {:>9} B  ns/dec {:>7.0} (first500 {:>6.0}, last500 {:>6.0})  allocs/dec steady {:>5.1} all {:>5.1}",
                name, m, lo, hi, c.mean, cost.live as f64, cost.blocks, bytes_rss_est, cost.peak, cost.ns_med, cost.ns_first500, cost.ns_last500, cost.allocs_per_dec_steady, cost.allocs_per_dec_all
            );
            rows.push(json!({
                "config": name, "bg_cpu_pct_before_timing": bg, "acc_full_mean": m, "acc_ci": [lo, hi], "acc_h250_mean": mean_sd(&h250).0,
                "diff_vs_guarded": c.mean, "diff_ci": [c.lo, c.hi],
                "state_live_bytes": cost.live, "state_blocks": cost.blocks, "state_bytes_incl_heap_hdr_est": bytes_rss_est, "peak_bytes": cost.peak,
                "ns_per_decision_median": cost.ns_med, "ns_per_decision_min": cost.ns_min, "ns_first500": cost.ns_first500, "ns_last500": cost.ns_last500,
                "allocs_per_decision_steady": cost.allocs_per_dec_steady, "allocs_per_decision_incl_growth": cost.allocs_per_dec_all,
            }));
        }
        // Pareto flags on (acc up, bytes down, ns down): dominated if another config is >= on all and > on one,
        // accuracy differences below the equivalence margin count as ties.
        let n = rows.len();
        let accm: Vec<f64> = rows.iter().map(|r| r["acc_full_mean"].as_f64().unwrap()).collect();
        let byt: Vec<f64> = rows.iter().map(|r| r["state_live_bytes"].as_f64().unwrap()).collect();
        let nsv: Vec<f64> = rows.iter().map(|r| r["ns_per_decision_min"].as_f64().unwrap()).collect();
        let mut flags = vec![];
        for i in 0..n {
            let mut dominated = false;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let acc_ge = accm[j] >= accm[i] - MARGIN;
                let better_somewhere = accm[j] > accm[i] + MARGIN || byt[j] < byt[i] * 0.98 || nsv[j] < nsv[i] * 0.98;
                if acc_ge && byt[j] <= byt[i] * 1.0001 && nsv[j] <= nsv[i] * 1.0001 && better_somewhere {
                    dominated = true;
                }
            }
            flags.push(!dominated);
        }
        println!("   Pareto-optimal (acc ties within {MARGIN}): {:?}", rows.iter().zip(&flags).filter(|(_, f)| **f).map(|(r, _)| r["config"].as_str().unwrap().to_string()).collect::<Vec<_>>());
        for (r, f) in rows.iter_mut().zip(&flags) {
            r["pareto_optimal"] = json!(f);
        }
        part1.push(json!({ "world": spec.label(), "agents": env0.n_agents(), "tasks": env0.len(), "distinct_features": distinct, "rows": rows }));
    }

    // ----------------------------------------------------------------------------------------
    // PART 2: representation search
    // ----------------------------------------------------------------------------------------
    println!("\n=== PART 2: smallest representation that keeps accuracy (margin {MARGIN}, Holm, TOST) ===");
    // --- tuning of the guard-count scale on seeds 1-8
    let mut tune_rows = vec![];
    let mut best_s = (f64::INFINITY, 1024u32);
    for s in [64u32, 256, 1024, 4096] {
        let scheme = sch("tune", IdMode::Table, q_maker(s));
        let mut dev = 0.0;
        let mut per_world = vec![];
        for spec in &worlds {
            let d: Vec<f64> = par_map(&tune, |&sd| {
                let env = spec.make(sd);
                let tb = Tables::new(&env);
                let mut a = mk_guarded(&env);
                let mut b = scheme.router(&env, 0);
                run_tab(b.as_mut(), &env, &tb, sd).full - run_tab(a.as_mut(), &env, &tb, sd).full
            });
            let m = d.iter().sum::<f64>() / d.len() as f64;
            dev += m.abs();
            per_world.push(m);
        }
        println!("[tune 1-8] guard-count scale 1/{s}: mean diff per world {:?}  sum|diff| {:.5}", per_world.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>(), dev);
        tune_rows.push(json!({"scale": s, "mean_diff_per_world": per_world, "sum_abs": dev}));
        if dev < best_s.0 - 1e-12 {
            best_s = (dev, s);
        }
    }
    let s_star = best_s.1;
    println!("[tune 1-8] frozen guard-count scale = 1/{s_star}");

    let mq = q_maker(s_star);
    let schemes: Vec<Scheme> = vec![
        sch("c f64/f64 (layout only)", IdMode::Table, mk::<f64, f64>),
        sch("c f32/f32", IdMode::Table, mk::<f32, f32>),
        sch("c u16/f32", IdMode::Table, mk::<U16, f32>),
        sch("c u16/q16 (frozen scale)", IdMode::Table, mq),
        sch("c u16/q16 + hash32 ids", IdMode::Hash32, mq),
        sch("c u16/q16 sparse rows", IdMode::Table, q_sparse(s_star)),
        sch("c f32/f32 sparse rows", IdMode::Table, mks::<f32, f32>),
        sch("c f16/q16 (binary16 weights)", IdMode::Table, mk::<F16, Q16<1024>>),
        sch("c bf16/q16 (bfloat16 weights)", IdMode::Table, mk::<Bf16, Q16<1024>>),
        sch("c f16/f16 (binary16 weights+counts)", IdMode::Table, mk::<F16, F16>),
        sch("c u8/q16 (u8 weights)", IdMode::Table, mk::<Fx8, Q16<1024>>),
        sch("c u16/q8 s=8 (u8 counts)", IdMode::Table, mk::<U16, Q8<8>>),
        sch("c u16/q16 + LRU 50% rows", IdMode::Table, mq).lim(Rows::Frac(0.5), Evict::Lru),
        sch("c u16/q16 + LRU 25% rows", IdMode::Table, mq).lim(Rows::Frac(0.25), Evict::Lru),
        sch("c u16/q16 + LFU 25% rows", IdMode::Table, mq).lim(Rows::Frac(0.25), Evict::Lfu { half: 600 }),
        sch("c u16/q16 + admit>=2", IdMode::Table, mq).adm(2),
        sch("c u16/q16 + admit>=2 + LRU 512 rows", IdMode::Table, mq).adm(2).lim(Rows::Abs(512), Evict::Lru),
        sch("c u16/q16 + admit>=2 + LRU 256 rows", IdMode::Table, mq).adm(2).lim(Rows::Abs(256), Evict::Lru),
        sch("c u16/q16 SPARSE + admit>=2 + LRU 512", IdMode::Table, q_sparse(s_star)).adm(2).lim(Rows::Abs(512), Evict::Lru),
    ];

    let eval_set = |schemes: &[Scheme], seed_set: &[u64], tag: &str| -> Vec<Value> {
        let mut out = vec![];
        // collect all (scheme, world) comparisons first, then Holm over the family
        let mut cmps: Vec<(usize, usize, Cmp, Cmp, Vec<f64>, Vec<f64>)> = vec![];
        for (si, scheme) in schemes.iter().enumerate() {
            for (wi, spec) in worlds.iter().enumerate() {
                let pairs: Vec<(RunStats, RunStats)> = par_map(seed_set, |&sd| {
                    let env = spec.make(sd);
                    let tb = Tables::new(&env);
                    let distinct = world_features(&env).len();
                    let mut a = mk_guarded(&env);
                    let mut b = scheme.router(&env, distinct);
                    (run_tab(a.as_mut(), &env, &tb, sd), run_tab(b.as_mut(), &env, &tb, sd))
                });
                let reff: Vec<f64> = pairs.iter().map(|p| p.0.full).collect();
                let schf: Vec<f64> = pairs.iter().map(|p| p.1.full).collect();
                let ref2: Vec<f64> = pairs.iter().map(|p| p.0.h250).collect();
                let sch2: Vec<f64> = pairs.iter().map(|p| p.1.h250).collect();
                cmps.push((si, wi, compare(&schf, &reff), compare(&sch2, &ref2), reff, schf));
            }
        }
        let p_eq: Vec<f64> = cmps.iter().map(|c| c.2.p_equiv).collect();
        let p_eq250: Vec<f64> = cmps.iter().map(|c| c.3.p_equiv).collect();
        let holm = holm_adjust(&p_eq);
        let holm250 = holm_adjust(&p_eq250);
        for (i, (si, wi, c, c250, _reff, _schf)) in cmps.iter().enumerate() {
            let v = verdict(c, holm[i]);
            out.push(json!({
                "set": tag, "scheme": schemes[*si].name, "world": worlds[*wi].label(),
                "full": cmp_json(c, Some(holm[i])), "h250": cmp_json(c250, Some(holm250[i])), "verdict": v.label,
            }));
        }
        out
    };

    let held_res = eval_set(&schemes, &held, "held-out");
    let confirm_res = eval_set(&schemes, &confirm, "confirm");
    let power_seeds = if quick { seeds(100, 139) } else { seeds(100, 399) };
    let power_schemes: Vec<Scheme> = schemes.iter().filter(|s| ["c f32/f32", "c u16/f32", "c f16/q16 (binary16 weights)", "c bf16/q16 (bfloat16 weights)", "c f16/f16 (binary16 weights+counts)", "c u8/q16 (u8 weights)", "c u16/q16 (frozen scale)", "c u16/q16 + hash32 ids", "c u16/q16 + admit>=2", "c u16/q16 + admit>=2 + LRU 512 rows"].contains(&s.name.as_str())).cloned().collect();
    let power_res = eval_set(&power_schemes, &power_seeds, "power");
    for (tag, res, sl) in [("held-out", &held_res, &schemes), ("confirm", &confirm_res, &schemes), ("power (300 extra seeds 100-399, equivalence-focused)", &power_res, &power_schemes)] {
        println!("\n{tag} seeds: diff of full-run expected success vs f64 crate guarded(hebr) [95% CI]  (verdict)");
        print!("{:<38}", "scheme");
        for w in &worlds {
            print!(" {:>34}", w.label());
        }
        println!();
        for s in sl.iter() {
            print!("{:<38}", s.name);
            for w in &worlds {
                let r = res.iter().find(|r| r["scheme"] == s.name.as_str() && r["world"] == w.label().as_str()).unwrap();
                let f = &r["full"];
                print!(" {:>+8.4} [{:+.4},{:+.4}] {:>11}", f["diff"].as_f64().unwrap(), f["ci_lo"].as_f64().unwrap(), f["ci_hi"].as_f64().unwrap(), r["verdict"].as_str().unwrap());
            }
            println!();
        }
    }

    // --- bytes of each scheme (single seed, allocator-measured, pre-interned ids)
    println!("\nstate bytes after {} tasks (allocator-measured live bytes after shrink-to-fit; per-brain only, shared vocab/kw excluded):", "run");
    let mut bytes_rows = vec![];
    // reference: crate guarded
    let mut bytes_table: Vec<(String, Vec<(i64, i64, usize)>)> = vec![];
    {
        let mut ref_row = vec![];
        for spec in &worlds {
            let env = spec.make(held[0]);
            let tb = Tables::new(&env);
            let c = cost_of(&mk_guarded, &env, &tb, held[0], 0);
            ref_row.push((c.live, c.blocks, 0usize));
        }
        bytes_table.push(("crate guarded(hebr) f64 (BTreeMap<String,..>)".into(), ref_row));
    }
    for scheme in &schemes {
        let mut row = vec![];
        for spec in &worlds {
            let env = spec.make(held[0]);
            let tb = Tables::new(&env);
            let p = pre(&env, scheme.mode);
            let cands: Vec<usize> = (0..tb.n).collect();
            let mut rng = Rng::new(held[0].wrapping_mul(1_000_003).wrapping_add(17));
            alloc_count::start();
            let mut b = (scheme.maker)(tb.n, p.kw.clone(), scheme.rows_budget(p.distinct));
            for t in 0..tb.len {
                let a = b.route(p.ids[t].as_slice(), &cands, &mut rng);
                b.feedback(p.ids[t].as_slice(), a, tb.out[t * tb.n + a] as f64);
            }
            let rows_n = b.rows();
            let ts = b.touched();
            let s_grown = alloc_count::snap();
            b.shrink();
            let s1 = alloc_count::stop();
            if scheme.name.contains("frozen") {
                println!("   [touched] {} / {}: rows {} mean agents-touched/row {:.2} (of {}) max {} frac<=4 {:.2}", scheme.name, spec.label(), ts.0, ts.1, tb.n, ts.2, ts.3);
            }
            std::hint::black_box(s_grown);
            row.push((s1.live, s1.blocks, rows_n));
            drop(b);
        }
        bytes_table.push((scheme.name.clone(), row));
    }
    print!("{:<46}", "scheme");
    for w in &worlds {
        print!(" {:>26}", w.label());
    }
    println!();
    for (name, row) in &bytes_table {
        print!("{:<46}", name);
        for (live, blocks, rows) in row {
            print!(" {:>10} B/{:>4} blk/{:>4}r", live, blocks, rows);
        }
        println!();
        bytes_rows.push(json!({"scheme": name, "per_world": row.iter().zip(&worlds).map(|((l, b, r), w)| json!({"world": w.label(), "live_bytes": l, "blocks": b, "rows": r})).collect::<Vec<_>>()}));
    }

    // ----------------------------------------------------------------------------------------
    // PART 3: agents per GiB (K real brains)
    // ----------------------------------------------------------------------------------------
    let part3 = part3(quick, s_star, held[0], blk_ov_mean);

    let out = json!({
        "tool": "f3_brain", "margin": MARGIN, "cores_logical": n_threads(), "quick": quick,
        "seeds": {"tune": [1, 8], "held": [held[0], held[held.len()-1]], "confirm": [confirm[0], confirm[confirm.len()-1]]},
        "heap_block_overhead_est_bytes": blk_ov_mean,
        "part1_pareto": part1, "part2_tune": tune_rows, "part2_frozen_scale": s_star,
        "part2_held": held_res, "part2_confirm": confirm_res, "part2_power": power_res, "part2_bytes": bytes_rows, "part3_agents_per_gib": part3,
        "runtime_s": t_start.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {out_path}  ({:.0}s)", t_start.elapsed().as_secs_f64());
}
