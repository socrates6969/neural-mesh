//! M1: does WARM-STARTING a router from a PRIOR SESSION's persisted journal (src/store.rs) beat COLD start on a later session
//! of the same world? And what is the ACTUAL benefit of SHARING journals across machines (diminishing returns, no assumed
//! linear compounding)?
//!
//! Protocol (pre-declared, nothing tuned): per seed s and world W, hidden structure H = W(s). Session 1 (stream seed s+1000+k) is
//! run by a cold router and journaled to a temp MeshStore; session 2 (stream seed s+5000, 2000 tasks, SAME hidden structure)
//! is run by (cold) vs (warm = fresh router + store.warm_replay). Metric: mean EXPECTED success of the chosen agent over the first H
//! tasks of session 2, H in {100,250,500}. Paired over seeds (warm - cold), 95% t-CI, Holm over all (world x router x H) tests per split.
//! Worlds: fixed (traps+drift), random, notraps (random hidden, no traps), STALE (session-2 hidden structure is a DIFFERENT random
//! world => the persisted evidence is wrong: the negative control / harm test).
//! Routers: hebbian-robust (defaults::recommended_inner) and recommended (Guarded keyword front + hebbian).
//! Parts: `warm` (N1 = 500), `dose` (N1 in 100,250,1000,2000, random world), `share` (M machines x 250 tasks merged).
//! Usage: m1_persist [--seeds 31-60] [--part warm|dose|share|all] [--out results/m1_persist.json]
use neural_mesh::defaults::{recommended_inner, recommended_router};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use neural_mesh::store::{MeshStore, Outcome};
use rayon::prelude::*;
use serde_json::{json, Value};

const N2: usize = 2000;
const HS: [usize; 3] = [100, 250, 500];

#[derive(Clone, Copy, PartialEq, Debug)]
enum Rk {
    Hebb,
    Rec,
}
fn rk_name(r: Rk) -> &'static str {
    match r {
        Rk::Hebb => "hebbian-robust",
        Rk::Rec => "recommended",
    }
}
fn mk_router(rk: Rk, w: &World) -> Box<dyn Router + Send> {
    match rk {
        Rk::Hebb => Box::new(HebbianRouter::new(6, recommended_inner())),
        Rk::Rec => Box::new(recommended_router(w.keyword_map_for(), 6)),
    }
}
fn scen(name: &str) -> Scenario {
    match name {
        "fixed" => Scenario::Fixed,
        "notraps" => Scenario::NoTraps,
        _ => Scenario::Random,
    }
}
fn cfg(name: &str, n: usize) -> WorldConfig {
    WorldConfig { n_tasks: n, scenario: scen(name), ..WorldConfig::default() }
}
static NAMES: [&str; 6] = ["a0", "a1", "a2", "a3", "a4", "a5"];
fn idx(n: &str) -> Option<usize> {
    NAMES.iter().position(|x| *x == n)
}

/// run a router over a world's stream; returns (expected success per task, outcomes journal)
fn run(world: &World, router: &mut dyn Router, n: usize, sess: &str, rng_seed: u64, ts0: u64) -> (Vec<f64>, Vec<Outcome>) {
    let mut rng = Rng::new(rng_seed);
    let cands: Vec<usize> = (0..6).collect();
    let mut ex = Vec::with_capacity(n);
    let mut js = Vec::with_capacity(n);
    for t in 0..n.min(world.len()) {
        let st = world.task(t);
        let a = router.select(&st.task, &cands, &mut rng);
        ex.push(world.success_prob(t, &[a]));
        let r = world.outcome(t, &[a]);
        router.update(&st.task, a, r);
        js.push(Outcome {
            id: format!("{sess}:{t}"),
            session: sess.into(),
            seq: t as u64,
            ts: ts0 + t as u64,
            world: "sim".into(),
            features: st.task.features.clone(),
            task_type: st.task.task_type.clone(),
            agent: NAMES[a].into(),
            reward: r,
        });
    }
    (ex, js)
}
fn meanh(ex: &[f64], h: usize) -> f64 {
    ex[..h.min(ex.len())].iter().sum::<f64>() / h.min(ex.len()) as f64
}

fn tmpstore(tag: &str) -> MeshStore {
    let p = std::env::temp_dir().join(format!("m1_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&p);
    MeshStore::open(p).unwrap()
}

/// one (seed, world, router, n1) trial -> [cold H100..500, warm H100..500]
fn trial(seed: u64, wname: &str, rk: Rk, n1: usize) -> [f64; 6] {
    let stale = wname == "stale";
    let base = if stale { "random" } else { wname };
    let w1 = World::new_stream(seed, seed + 1000, cfg(base, n1.max(1)));
    let hid2 = if stale { seed + 777 } else { seed };
    let w2 = World::new_stream(hid2, seed + 5000, cfg(base, N2));
    let st = tmpstore(&format!("{seed}_{wname}_{}_{n1}", rk_name(rk)));
    let mut r1 = mk_router(rk, &w1);
    let (_, js) = run(&w1, r1.as_mut(), n1, "s1", seed * 31 + 1, 0);
    st.append_outcomes(&js).unwrap();
    let mut rc = mk_router(rk, &w2);
    let (exc, _) = run(&w2, rc.as_mut(), N2, "s2c", seed * 31 + 2, 10_000);
    let mut rw = mk_router(rk, &w2);
    st.warm_replay(rw.as_mut(), &idx, None, None).unwrap();
    let (exw, _) = run(&w2, rw.as_mut(), N2, "s2w", seed * 31 + 2, 10_000);
    let _ = std::fs::remove_dir_all(st.root());
    let mut o = [0.0; 6];
    for (i, h) in HS.iter().enumerate() {
        o[i] = meanh(&exc, *h);
        o[3 + i] = meanh(&exw, *h);
    }
    o
}

struct Row {
    world: String,
    router: String,
    h: usize,
    label: String,
    diffs: Vec<f64>,
    cold: f64,
    warm: f64,
}

fn rows_for(seeds: &[u64], worlds: &[&str], rks: &[Rk], n1: usize, tag: &str) -> Vec<Row> {
    let mut rows = vec![];
    for w in worlds {
        for rk in rks {
            let res: Vec<[f64; 6]> = seeds.par_iter().map(|&s| trial(s, w, *rk, n1)).collect();
            for (i, h) in HS.iter().enumerate() {
                let c: Vec<f64> = res.iter().map(|r| r[i]).collect();
                let wv: Vec<f64> = res.iter().map(|r| r[3 + i]).collect();
                rows.push(Row {
                    world: w.to_string(),
                    router: rk_name(*rk).into(),
                    h: *h,
                    label: tag.into(),
                    diffs: wv.iter().zip(&c).map(|(a, b)| a - b).collect(),
                    cold: c.iter().sum::<f64>() / c.len() as f64,
                    warm: wv.iter().sum::<f64>() / wv.len() as f64,
                });
            }
        }
    }
    rows
}

fn report(rows: &[Row], zero: &[f64], title: &str) -> Value {
    let ps: Vec<f64> = rows.iter().map(|r| paired_t_pvalue(&r.diffs, &zero[..r.diffs.len()])).collect();
    let adj = holm_adjust(&ps);
    println!("--- {title}");
    let mut out = vec![];
    for (r, ph) in rows.iter().zip(adj) {
        let pd = paired_diff(&r.diffs, &zero[..r.diffs.len()]);
        println!(
            "  {:8} {:15} {:7} H{:<4} cold {:.4} warm {:.4} d {:+.4} [{:+.4},{:+.4}] W/L {}/{} Holm p {:.2e}",
            r.world, r.router, r.label, r.h, r.cold, r.warm, pd.diff.mean, pd.diff.lo, pd.diff.hi, pd.wins, pd.losses, ph
        );
        out.push(json!({"world": r.world, "router": r.router, "h": r.h, "label": r.label, "cold": r.cold, "warm": r.warm, "d": pd.diff.mean, "lo": pd.diff.lo, "hi": pd.diff.hi, "wins": pd.wins, "losses": pd.losses, "p_holm": ph}));
    }
    Value::Array(out)
}

fn parse_range(s: &str) -> Vec<u64> {
    let (a, b) = s.split_once('-').unwrap();
    (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
}


/// Staleness detector around a warm router: compares observed rewards with what the journal predicts for the chosen agent
/// (mean reward per (feature, agent) from the replayed journal). If the cumulative surprise D = sum(actual - pred) falls below
/// -z*sqrt(n*0.25) after `min_n` informative outcomes, the persisted evidence is declared stale and the router is reset to COLD.
struct WarmGuard {
    inner: Box<dyn Router + Send>,
    fresh: Box<dyn Fn() -> Box<dyn Router + Send> + Send>,
    table: std::collections::HashMap<(String, usize), (f64, f64)>,
    d: f64,
    n: f64,
    z: f64,
    min_n: f64,
    tripped: bool,
}
impl Router for WarmGuard {
    fn name(&self) -> String {
        "warm-guard".into()
    }
    fn select(&mut self, task: &neural_mesh::router::Task, c: &[usize], rng: &mut Rng) -> usize {
        self.inner.select(task, c, rng)
    }
    fn update(&mut self, task: &neural_mesh::router::Task, agent: usize, reward: f64) {
        if !self.tripped {
            let mut s = 0.0;
            let mut k = 0.0;
            for f in &task.features {
                if let Some((sum, cnt)) = self.table.get(&(f.clone(), agent)) {
                    if *cnt >= 3.0 {
                        s += sum / cnt;
                        k += 1.0;
                    }
                }
            }
            if k > 0.0 {
                self.d += reward - s / k;
                self.n += 1.0;
                if self.n >= self.min_n && self.d < -self.z * (self.n * 0.25).sqrt() {
                    self.tripped = true;
                    self.inner = (self.fresh)();
                }
            }
        }
        self.inner.update(task, agent, reward);
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Full,
    Recent(usize),
    Guard { z: f64, min_n: usize },
    GuardRecent { z: f64, min_n: usize, k: usize },
    InnerOnly,
}
fn mode_name(m: Mode) -> String {
    match m {
        Mode::Full => "full".into(),
        Mode::Recent(k) => format!("recent{k}"),
        Mode::Guard { z, min_n } => format!("guard(z{z},n{min_n})"),
        Mode::GuardRecent { z, min_n, k } => format!("guard(z{z},n{min_n})+recent{k}"),
        Mode::InnerOnly => "inner-only".into(),
    }
}
fn build_warm(rk: Rk, w2: &World, st: &MeshStore, mode: Mode) -> Box<dyn Router + Send> {
    if let (Mode::InnerOnly, Rk::Rec) = (mode, rk) {
        // R34: replay only into the inner learner; the keyword gate keeps cold counts
        let mut g = recommended_router(w2.keyword_map_for(), 6);
        st.warm_replay(&mut g.inner, &idx, None, None).unwrap();
        return Box::new(g);
    }
    let mut rw = mk_router(rk, w2);
    let recent = match mode {
        Mode::Recent(k) => Some(k),
        Mode::GuardRecent { k, .. } => Some(k),
        _ => None,
    };
    st.warm_replay(rw.as_mut(), &idx, None, recent).unwrap();
    match mode {
        Mode::Guard { z, min_n } | Mode::GuardRecent { z, min_n, .. } => {
            let mut table = std::collections::HashMap::new();
            for o in st.outcomes().unwrap() {
                if let Some(a) = idx(&o.agent) {
                    for f in &o.features {
                        let e: &mut (f64, f64) = table.entry((f.clone(), a)).or_insert((0.0, 0.0));
                        e.0 += o.reward;
                        e.1 += 1.0;
                    }
                }
            }
            let kw = w2.keyword_map_for();
            let fresh: Box<dyn Fn() -> Box<dyn Router + Send> + Send> = Box::new(move || match rk {
                Rk::Hebb => Box::new(HebbianRouter::new(6, recommended_inner())),
                Rk::Rec => Box::new(recommended_router(kw.clone(), 6)),
            });
            Box::new(WarmGuard { inner: rw, fresh, table, d: 0.0, n: 0.0, z, min_n: min_n as f64, tripped: false })
        }
        _ => rw,
    }
}
fn mit_trial(seed: u64, wname: &str, rk: Rk, n1: usize, modes: &[Mode]) -> Vec<[f64; 3]> {
    let stale = wname == "stale";
    let base = if stale { "random" } else { wname };
    let w1 = World::new_stream(seed, seed + 1000, cfg(base, n1));
    let w2 = World::new_stream(if stale { seed + 777 } else { seed }, seed + 5000, cfg(base, N2));
    let st = tmpstore(&format!("mit_{seed}_{wname}_{}_{n1}", rk_name(rk)));
    let mut r1 = mk_router(rk, &w1);
    let (_, js) = run(&w1, r1.as_mut(), n1, "s1", seed * 31 + 1, 0);
    st.append_outcomes(&js).unwrap();
    let mut out = vec![];
    let mut rc = mk_router(rk, &w2);
    let (exc, _) = run(&w2, rc.as_mut(), N2, "c", seed * 31 + 2, 10_000);
    out.push([meanh(&exc, 100), meanh(&exc, 250), meanh(&exc, 500)]);
    for m in modes {
        let mut rw = build_warm(rk, &w2, &st, *m);
        let (ex, _) = run(&w2, rw.as_mut(), N2, "w", seed * 31 + 2, 10_000);
        out.push([meanh(&ex, 100), meanh(&ex, 250), meanh(&ex, 500)]);
    }
    let _ = std::fs::remove_dir_all(st.root());
    out
}


/// REAL durable store check: populate (idempotent) the journal at `root` with synthetic Fixed-world sessions (streams 9000+k), then
/// compare session-2 cold vs warm-from-the-durable-store on seeds disjoint from the journal's streams.
fn real_store(root: &str, seeds: &[u64], zero: &[f64]) -> Value {
    let st = MeshStore::open(root).unwrap();
    let base = World::new_stream(1, 9000, cfg("fixed", 500));
    let names = base.agent_names();
    for k in 0..5u64 {
        let w = World::new_stream(1, 9000 + k, cfg("fixed", 500));
        let mut r = mk_router(Rk::Rec, &w);
        let (_, mut js) = run(&w, r.as_mut(), 500, &format!("populate-{k}"), 9000 + k, 100 * k);
        for (o, _) in js.iter_mut().zip(0..) {
            let a = idx(&o.agent).unwrap();
            o.agent = names[a].clone();
            o.world = "sim:fixed".into();
        }
        st.append_outcomes(&js).unwrap();
    }
    println!("durable store {root}: {:?}", st.stats().unwrap());
    let nm = names.clone();
    let idx_real = move |n: &str| nm.iter().position(|x| x == n);
    let mut rows = vec![];
    for rk in [Rk::Hebb, Rk::Rec] {
        let res: Vec<[f64; 6]> = seeds
            .par_iter()
            .map(|&s| {
                let w2 = World::new_stream(1, s + 5000, cfg("fixed", N2));
                let mut rc = mk_router(rk, &w2);
                let (exc, _) = run(&w2, rc.as_mut(), N2, "c", s * 31 + 2, 10_000);
                let mut rw = mk_router(rk, &w2);
                st.warm_replay(rw.as_mut(), &idx_real, Some("sim:fixed"), None).unwrap();
                let (exw, _) = run(&w2, rw.as_mut(), N2, "w", s * 31 + 2, 10_000);
                [meanh(&exc, 100), meanh(&exc, 250), meanh(&exc, 500), meanh(&exw, 100), meanh(&exw, 250), meanh(&exw, 500)]
            })
            .collect();
        for (i, h) in HS.iter().enumerate() {
            let c: Vec<f64> = res.iter().map(|r| r[i]).collect();
            let wv: Vec<f64> = res.iter().map(|r| r[3 + i]).collect();
            rows.push(Row { world: "sim:fixed".into(), router: rk_name(rk).into(), h: *h, label: "durable".into(), diffs: wv.iter().zip(&c).map(|(a, b)| a - b).collect(), cold: c.iter().sum::<f64>() / c.len() as f64, warm: wv.iter().sum::<f64>() / wv.len() as f64 });
        }
    }
    report(&rows, zero, "REAL durable store (2500 journaled outcomes) warm vs cold")
}

/// sharing: M machines each run `per` tasks cold in their own stream; journals merged; session-2 warm from (own only) vs (merged)
fn share_trial(seed: u64, wname: &str, rk: Rk, m: usize, per: usize) -> [f64; 6] {
    let w2 = World::new_stream(seed, seed + 5000, cfg(wname, N2));
    let root = tmpstore(&format!("sh_{seed}_{wname}_{}_{m}", rk_name(rk)));
    let mut own: Option<MeshStore> = None;
    for k in 0..m {
        let wk = World::new_stream(seed, seed + 1000 + 17 * k as u64, cfg(wname, per));
        let mut r = mk_router(rk, &wk);
        let (_, js) = run(&wk, r.as_mut(), per, &format!("m{k}"), seed * 31 + 7 + k as u64, 0);
        let s = tmpstore(&format!("shm_{seed}_{wname}_{}_{m}_{k}", rk_name(rk)));
        s.append_outcomes(&js).unwrap();
        root.merge_from(&s).unwrap();
        if k == 0 {
            own = Some(s);
        } else {
            let _ = std::fs::remove_dir_all(s.root());
        }
    }
    let own = own.unwrap();
    let mut rc = mk_router(rk, &w2);
    let (exc, _) = run(&w2, rc.as_mut(), N2, "c", seed * 31 + 2, 10_000);
    let mut ro = mk_router(rk, &w2);
    own.warm_replay(ro.as_mut(), &idx, None, None).unwrap();
    let (exo, _) = run(&w2, ro.as_mut(), N2, "o", seed * 31 + 2, 10_000);
    let mut rs = mk_router(rk, &w2);
    root.warm_replay(rs.as_mut(), &idx, None, None).unwrap();
    let (exs, _) = run(&w2, rs.as_mut(), N2, "s", seed * 31 + 2, 10_000);
    let _ = std::fs::remove_dir_all(root.root());
    let _ = std::fs::remove_dir_all(own.root());
    [meanh(&exc, 100), meanh(&exc, 250), meanh(&exo, 100), meanh(&exo, 250), meanh(&exs, 100), meanh(&exs, 250)]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone()).unwrap_or(d.into());
    let seeds = parse_range(&get("--seeds", "31-60"));
    let part = get("--part", "all");
    let out_path = get("--out", "results/m1_persist.json");
    let zero = vec![0.0; seeds.len()];
    let mut j = serde_json::Map::new();
    if part == "real" {
        j.insert("real".into(), real_store(&get("--store", "C:/Users/mariu/experiments/mesh-memory"), &seeds, &zero));
    }
    if part == "mit" {
        let modes = [Mode::Full, Mode::InnerOnly, Mode::Recent(100), Mode::Recent(250), Mode::Recent(500), Mode::Recent(1000), Mode::Guard { z: 1.64, min_n: 30 }, Mode::Guard { z: 1.0, min_n: 20 }, Mode::Guard { z: 2.33, min_n: 40 }, Mode::GuardRecent { z: 1.64, min_n: 30, k: 250 }];
        let mut rows = vec![];
        for w in ["fixed", "random", "notraps", "stale"] {
            for rk in [Rk::Hebb, Rk::Rec] {
                let res: Vec<Vec<[f64; 3]>> = seeds.par_iter().map(|&s| mit_trial(s, w, rk, std::env::var("M1_N1").ok().and_then(|x| x.parse().ok()).unwrap_or(500), &modes)).collect();
                for (mi, m) in modes.iter().enumerate() {
                    for (hi, h) in HS.iter().enumerate() {
                        let c: Vec<f64> = res.iter().map(|r| r[0][hi]).collect();
                        let wv: Vec<f64> = res.iter().map(|r| r[mi + 1][hi]).collect();
                        rows.push(Row { world: w.into(), router: rk_name(rk).into(), h: *h, label: mode_name(*m), diffs: wv.iter().zip(&c).map(|(a, b)| a - b).collect(), cold: c.iter().sum::<f64>() / c.len() as f64, warm: wv.iter().sum::<f64>() / wv.len() as f64 });
                    }
                }
            }
        }
        j.insert("mit".into(), report(&rows, &zero, "MITIGATIONS (warm mode - cold), N1=500"));
    }
    if part == "warm" || part == "all" {
        let rows = rows_for(&seeds, &["fixed", "random", "notraps", "stale"], &[Rk::Hebb, Rk::Rec], 500, "N1=500");
        j.insert("warm".into(), report(&rows, &zero, "WARM (N1=500) vs COLD, session-2 expected success"));
    }
    if part == "dose" || part == "all" {
        let mut rows = vec![];
        for n1 in [100usize, 250, 1000, 2000] {
            rows.extend(rows_for(&seeds, &["random"], &[Rk::Hebb, Rk::Rec], n1, &format!("N1={n1}")));
        }
        j.insert("dose".into(), report(&rows, &zero, "DOSE (random world)"));
    }
    if part == "share" || part == "all" {
        println!("--- SHARE: M machines x 250 cold tasks; session-2 H100 mean expected success (random world)");
        let mut out = vec![];
        for rk in [Rk::Hebb, Rk::Rec] {
            for m in [1usize, 2, 4, 8] {
                let res: Vec<[f64; 6]> = seeds.par_iter().map(|&s| share_trial(s, "random", rk, m, 250)).collect();
                let mean = |i: usize| res.iter().map(|r| r[i]).sum::<f64>() / res.len() as f64;
                let dsh: Vec<f64> = res.iter().map(|r| r[4] - r[2]).collect();
                let pd = paired_diff(&dsh, &zero);
                let dsc: Vec<f64> = res.iter().map(|r| r[4] - r[0]).collect();
                println!(
                    "  {:15} M={m}: cold {:.4} | own(250) {:.4} | shared({}) {:.4}  shared-own {:+.4} [{:+.4},{:+.4}]  shared-cold {:+.4}",
                    rk_name(rk), mean(0), mean(2), m * 250, mean(4), pd.diff.mean, pd.diff.lo, pd.diff.hi, paired_diff(&dsc, &zero).diff.mean
                );
                out.push(json!({"router": rk_name(rk), "m": m, "cold100": mean(0), "own100": mean(2), "shared100": mean(4), "cold250": mean(1), "own250": mean(3), "shared250": mean(5), "shared_minus_own100": pd.diff.mean, "lo": pd.diff.lo, "hi": pd.diff.hi}));
            }
        }
        j.insert("share".into(), Value::Array(out));
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&Value::Object(j)).unwrap()).unwrap();
}
