//! D5 (code-quality worker): does the router make real task work higher QUALITY?
//!
//! Synthetic multi-domain task world (coding / math / STEM / debugging, 16
//! sub-skills with keywords), GRADED quality in [0,1], HIDDEN per-agent per-sub-skill
//! competence (specialists, generalists, noisy, failing/crashing agents), drift,
//! misleading keyword hints, mixed-domain tasks. 7 world variants.
//!
//! Routers: keyword, eps-greedy(tuned), pooled-thompson(tuned), hebbian-robust,
//! warm(hebbian-robust), guarded(hebr) frozen from x1 (tau .55 z .5 kw(2,1) disc .97),
//! plus guarded(hebr) re-tuned here on seeds 1-8.
//! Protocol: tune seeds 1-8, held-out 31-60 (default) / confirm 61-90 via --held,
//! paired 95% t-CI vs best fair baseline (by held-out mean) per (world, window),
//! Holm over candidate x world x window.
//! Usage: cargo run --release --bin d5_quality -- [--held 31-60] [--out results/d5_quality.json]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

const N_TASKS: usize = 2000;
const NS: usize = 16; // sub-skills
const NA: usize = 8; // agents
const DOMS: [&str; 4] = ["coding", "math", "stem", "debugging"];
const WINDOWS: [&str; 4] = ["first100", "first250", "all", "last500"];
const SUB: [(&str, [&str; 3]); NS] = [
    ("rust", ["rust", "cargo", "borrowck"]),
    ("python", ["python", "pip", "pandas"]),
    ("javascript", ["javascript", "npm", "typescript"]),
    ("sql", ["sql", "postgres", "join"]),
    ("algebra", ["algebra", "polynomial", "equation"]),
    ("proof", ["proof", "lemma", "induction"]),
    ("calculus", ["calculus", "integral", "derivative"]),
    ("statistics", ["statistics", "regression", "variance"]),
    ("physics", ["physics", "momentum", "thermo"]),
    ("chemistry", ["chemistry", "reaction", "molar"]),
    ("biology", ["biology", "protein", "genome"]),
    ("circuits", ["circuits", "voltage", "resistor"]),
    ("stack-trace", ["stack-trace", "panic", "traceback"]),
    ("segfault", ["segfault", "core-dump", "null-ptr"]),
    ("race-condition", ["race", "deadlock", "mutex"]),
    ("memory-leak", ["leak", "heap", "valgrind"]),
];
const DOMKW: [&str; 4] = ["code", "math", "science", "debug"];
const GENERIC: [&str; 6] = ["fix", "help", "urgent", "please", "explain", "review"];

#[derive(Clone, Copy, Debug)]
enum WK {
    Clear,
    Generalists,
    Misleading,
    Drift,
    NoisyFailing,
    Mixed,
    Control,
}
#[derive(Clone, Copy, Debug)]
struct WorldSpec {
    name: &'static str,
    kind: WK,
    mixed_p: f64,
    decoy_p: f64,
    mislabel_p: f64,
    trap_frac: f64,
    map_err: f64,
}
fn worlds() -> Vec<WorldSpec> {
    let b = WorldSpec { name: "", kind: WK::Clear, mixed_p: 0.0, decoy_p: 0.0, mislabel_p: 0.0, trap_frac: 0.0, map_err: 0.05 };
    vec![
        WorldSpec { name: "clear-specialists", kind: WK::Clear, ..b },
        WorldSpec { name: "many-generalists", kind: WK::Generalists, map_err: 0.1, ..b },
        WorldSpec { name: "misleading-keywords", kind: WK::Misleading, decoy_p: 0.3, mislabel_p: 0.15, trap_frac: 0.3, ..b },
        WorldSpec { name: "drift", kind: WK::Drift, ..b },
        WorldSpec { name: "noisy+failing", kind: WK::NoisyFailing, ..b },
        WorldSpec { name: "mixed-domain", kind: WK::Mixed, mixed_p: 0.6, ..b },
        WorldSpec { name: "control(equal)", kind: WK::Control, map_err: 0.2, ..b },
    ]
}

struct World {
    tasks: Vec<Task>,
    exp: Vec<[f64; NA]>,
    real: Vec<[f64; NA]>,
    prim: Vec<usize>,
    kwmap: HashMap<String, usize>,
}

fn gauss(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}
fn cl(x: f64) -> f64 {
    x.clamp(0.02, 0.99)
}

struct Agents {
    s0: Vec<[f64; NS]>,
    s1: Vec<[f64; NS]>,
    nominal: Vec<[f64; NS]>,
    sd: [f64; NA],
    crash: [f64; NA],
}

fn build_agents(spec: &WorldSpec, rng: &mut Rng) -> Agents {
    let mut s = vec![[0.0f64; NS]; NA];
    let mut sd = [0.08f64; NA];
    let mut crash = [0.0f64; NA];
    let mut perm: Vec<usize> = (0..6).collect();
    shuffle(&mut perm, rng);
    let specialist = |s: &mut Vec<[f64; NS]>, rng: &mut Rng, nown: usize, owner: &dyn Fn(usize) -> usize| {
        for a in 0..nown {
            let owned: Vec<usize> = (0..NS).filter(|&x| owner(x) == a).collect();
            for x in 0..NS {
                s[a][x] = if owned.contains(&x) {
                    0.88 + 0.04 * gauss(rng)
                } else if owned.iter().any(|&o| o / 4 == x / 4) {
                    0.40 + 0.08 * rng.f64()
                } else {
                    0.28 + 0.08 * rng.f64()
                };
            }
        }
    };
    let mut nominal: Option<Vec<[f64; NS]>> = None;
    match spec.kind {
        WK::Clear | WK::Misleading | WK::Drift | WK::Mixed => {
            specialist(&mut s, rng, 6, &|x| perm[x % 6]);
            let gl = if matches!(spec.kind, WK::Mixed) { 0.72 } else { 0.58 };
            for a in 6..8 {
                for x in 0..NS {
                    s[a][x] = gl + 0.05 * gauss(rng);
                }
            }
        }
        WK::Generalists => {
            for a in 0..6 {
                let b = 0.62 + 0.16 * rng.f64();
                for x in 0..NS {
                    s[a][x] = b + 0.07 * gauss(rng);
                }
            }
            let mut subs: Vec<usize> = (0..NS).collect();
            shuffle(&mut subs, rng);
            for a in 6..8 {
                for x in 0..NS {
                    s[a][x] = 0.6 + 0.04 * gauss(rng);
                }
                for j in 0..2 {
                    s[a][subs[(a - 6) * 2 + j]] = 0.88;
                }
            }
            sd = [0.1; NA];
        }
        WK::NoisyFailing => {
            let mut p4: Vec<usize> = (0..4).collect();
            shuffle(&mut p4, rng);
            specialist(&mut s, rng, 4, &|x| p4[x % 4]);
            for a in 0..4 {
                for x in 0..NS {
                    if s[a][x] > 0.8 {
                        s[a][x] = 0.82 + 0.03 * gauss(rng);
                    }
                }
            }
            let mut subs: Vec<usize> = (0..NS).collect();
            shuffle(&mut subs, rng);
            let mut nom = s.clone();
            for a in 4..6 {
                for x in 0..NS {
                    s[a][x] = 0.30 + 0.05 * rng.f64();
                    nom[a][x] = s[a][x];
                }
                for j in 0..3 {
                    s[a][subs[(a - 4) * 3 + j]] = 0.93;
                    nom[a][subs[(a - 4) * 3 + j]] = 0.93;
                }
                crash[a] = 0.5;
            }
            for a in 6..8 {
                for x in 0..NS {
                    s[a][x] = 0.68 + 0.05 * gauss(rng);
                    nom[a][x] = s[a][x];
                }
                sd[a] = 0.3;
            }
            nominal = Some(nom);
        }
        WK::Control => {
            for a in 0..NA {
                for x in 0..NS {
                    s[a][x] = 0.7 + 0.02 * gauss(rng);
                }
            }
            sd = [0.1; NA];
        }
    }
    for a in 0..NA {
        for x in 0..NS {
            s[a][x] = cl(s[a][x]);
        }
    }
    let mut s1 = s.clone();
    if matches!(spec.kind, WK::Drift) {
        let mut sp: Vec<usize> = (0..6).collect();
        shuffle(&mut sp, rng);
        for &a in sp.iter().take(3) {
            for x in 0..NS {
                if s[a][x] > 0.8 {
                    s1[a][x] = 0.43;
                    let mut b = rng.below(NA);
                    while b == a {
                        b = rng.below(NA);
                    }
                    s1[b][x] = 0.9;
                }
            }
        }
    }
    let nominal = nominal.unwrap_or_else(|| s.clone());
    Agents { s0: s, s1, nominal, sd, crash }
}

fn argmax(xs: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[b] {
            b = i;
        }
    }
    b
}

fn gen_world(seed: u64, wi: usize, spec: &WorldSpec) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(wi as u64 * 101 + 3));
    let ag = build_agents(spec, &mut rng);
    // keyword map (advertised skills, possibly wrong)
    let mut ad = [0usize; NS];
    for x in 0..NS {
        let col: Vec<f64> = (0..NA).map(|a| ag.nominal[a][x]).collect();
        ad[x] = argmax(&col);
        if rng.f64() < spec.map_err {
            ad[x] = rng.below(NA);
        }
    }
    if spec.trap_frac > 0.0 {
        let mut subs: Vec<usize> = (0..NS).collect();
        shuffle(&mut subs, &mut rng);
        let nt = (spec.trap_frac * NS as f64).round() as usize;
        for &x in subs.iter().take(nt) {
            let mut order: Vec<usize> = (0..NA).collect();
            order.sort_by(|&p, &q| ag.s0[p][x].partial_cmp(&ag.s0[q][x]).unwrap());
            ad[x] = order[rng.below(NA / 2)];
        }
    }
    let mut kwmap = HashMap::new();
    for x in 0..NS {
        for k in SUB[x].1 {
            kwmap.insert(k.to_string(), ad[x]);
        }
    }
    for d in 0..4 {
        let m: Vec<f64> = (0..NA).map(|a| (0..4).map(|j| ag.nominal[a][d * 4 + j]).sum::<f64>()).collect();
        kwmap.insert(DOMKW[d].to_string(), argmax(&m));
    }
    let tdrift = N_TASKS / 2;
    let (mut tasks, mut exp, mut real, mut prim) = (vec![], vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let s1 = rng.below(NS);
        let mixed = rng.bernoulli(spec.mixed_p);
        let s2 = if mixed {
            loop {
                let c = rng.below(NS);
                if c / 4 != s1 / 4 {
                    break Some(c);
                }
            }
        } else {
            None
        };
        let diff = rng.f64();
        let mut f: Vec<String> = vec![];
        // visible primary keywords (maybe mislabelled)
        let shown1 = if rng.bernoulli(spec.mislabel_p) { rng.below(NS) } else { s1 };
        let pick = |x: usize, f: &mut Vec<String>, rng: &mut Rng| {
            let before = f.len();
            for k in SUB[x].1 {
                if rng.bernoulli(0.55) {
                    f.push(k.to_string());
                }
            }
            if f.len() == before && rng.bernoulli(0.85) {
                f.push(SUB[x].1[rng.below(3)].to_string());
            }
        };
        pick(shown1, &mut f, &mut rng);
        if let Some(c) = s2 {
            pick(c, &mut f, &mut rng);
        }
        if rng.bernoulli(0.4) {
            f.push(DOMKW[s1 / 4].to_string());
        }
        // debugging tasks usually mention a language (realistic decoy), all worlds
        if s1 / 4 == 3 && rng.bernoulli(0.35) {
            f.push(SUB[rng.below(4)].1[rng.below(3)].to_string());
        }
        if rng.bernoulli(spec.decoy_p) {
            let o = rng.below(NS);
            f.push(SUB[o].1[rng.below(3)].to_string());
        }
        for _ in 0..2 {
            if rng.bernoulli(0.3) {
                f.push(GENERIC[rng.below(6)].to_string());
            }
        }
        f.dedup();
        let sk = if i >= tdrift { &ag.s1 } else { &ag.s0 };
        let (w1, w2) = if s2.is_some() { (0.65, 0.35) } else { (1.0, 0.0) };
        let mut e = [0.0; NA];
        let mut r = [0.0; NA];
        for a in 0..NA {
            let base = w1 * sk[a][s1] + s2.map(|c| w2 * sk[a][c]).unwrap_or(0.0);
            let q = cl(base * (1.0 - 0.3 * diff));
            e[a] = (1.0 - ag.crash[a]) * q + ag.crash[a] * 0.05;
            let cu = rng.f64();
            let z = gauss(&mut rng);
            r[a] = if cu < ag.crash[a] { 0.05 } else { (q + ag.sd[a] * z).clamp(0.0, 1.0) };
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        exp.push(e);
        real.push(r);
        prim.push(s1 / 4);
    }
    World { tasks, exp, real, prim, kwmap }
}

// ------------------------------------------------------------------ routers
fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    WarmHebR,
    GuardHebR(GuardConfig),
}

#[derive(Clone, Debug, Default)]
struct Out {
    q: [f64; 4],        // first100, first250, all, last500 (mean expected quality)
    regret: [f64; 4],   // oracle - q
    best_all: f64,      // fraction best agent chosen
    best_last: f64,
    q_real: f64,        // mean realized quality
    dq: [f64; 4],       // per-domain mean expected quality
    dbest: [f64; 4],    // per-domain best-agent fraction
    doracle: [f64; 4],
}

fn drive<R: Router>(mut r: R, w: &World, seed: u64) -> Out {
    let cands: Vec<usize> = (0..NA).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut o = Out::default();
    let mut sums = [0.0f64; 4];
    let mut osum = [0.0f64; 4];
    let (mut dq, mut dn, mut db, mut dor) = ([0.0f64; 4], [0.0f64; 4], [0.0f64; 4], [0.0f64; 4]);
    let (mut b_all, mut b_last, mut real) = (0.0, 0.0, 0.0);
    for (i, task) in w.tasks.iter().enumerate() {
        let a = r.select(task, &cands, &mut rng);
        let e = &w.exp[i];
        let bi = argmax(e);
        let orc = e[bi];
        let q = e[a];
        let hit = if a == bi || (orc - q) < 1e-9 { 1.0 } else { 0.0 };
        if i < 100 {
            sums[0] += q;
            osum[0] += orc;
        }
        if i < 250 {
            sums[1] += q;
            osum[1] += orc;
        }
        sums[2] += q;
        osum[2] += orc;
        if i >= N_TASKS - 500 {
            sums[3] += q;
            osum[3] += orc;
            b_last += hit;
        }
        b_all += hit;
        real += w.real[i][a];
        let d = w.prim[i];
        dq[d] += q;
        dn[d] += 1.0;
        db[d] += hit;
        dor[d] += orc;
        r.update(task, a, w.real[i][a]);
    }
    let n = [100.0, 250.0, N_TASKS as f64, 500.0];
    for k in 0..4 {
        o.q[k] = sums[k] / n[k];
        o.regret[k] = osum[k] / n[k] - o.q[k];
    }
    o.best_all = b_all / N_TASKS as f64;
    o.best_last = b_last / 500.0;
    o.q_real = real / N_TASKS as f64;
    for d in 0..4 {
        o.dq[d] = dq[d] / dn[d].max(1.0);
        o.dbest[d] = db[d] / dn[d].max(1.0);
        o.doracle[d] = dor[d] / dn[d].max(1.0);
    }
    o
}

fn run(kind: &Kind, w: &World, seed: u64) -> Out {
    let km = || w.kwmap.clone();
    match kind {
        Kind::Keyword => drive(KeywordRouter::new(km(), NA), w, seed),
        Kind::Eps(g, e) => drive(EpsGreedyRouter::new(NA, *g, *e), w, seed),
        Kind::Pts(g) => drive(PooledThompsonRouter::new(NA, *g), w, seed),
        Kind::HebR => drive(HebbianRouter::new(NA, heb_robust_cfg()), w, seed),
        Kind::WarmHebR => {
            drive(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), km(), NA, 1.0).with_evidence(Evidence::MeanFeature), w, seed)
        }
        Kind::GuardHebR(g) => drive(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), km(), NA, *g), w, seed),
    }
}

/// results[kind][world][seed]
fn eval(kinds: &[Kind], seeds: &[u64], ws: &[WorldSpec]) -> Vec<Vec<Vec<Out>>> {
    let per_seed: Vec<Vec<Vec<Out>>> = seeds
        .par_iter()
        .map(|&sd| {
            ws.iter()
                .enumerate()
                .map(|(wi, s)| {
                    let w = gen_world(sd, wi, s);
                    kinds.iter().map(|k| run(k, &w, sd)).collect::<Vec<Out>>()
                })
                .collect()
        })
        .collect();
    (0..kinds.len())
        .map(|k| (0..ws.len()).map(|wi| per_seed.iter().map(|ps| ps[wi][k].clone()).collect()).collect())
        .collect()
}

fn tune_obj(r: &[Vec<Out>]) -> f64 {
    // mean over worlds & seeds of mean(first100, first250, all, last500) quality
    let mut t = 0.0;
    let mut n = 0.0;
    for w in r {
        for o in w {
            t += o.q.iter().sum::<f64>() / 4.0;
            n += 1.0;
        }
    }
    t / n
}

fn tune(label: &str, cands: Vec<Kind>, seeds: &[u64], ws: &[WorldSpec]) -> Kind {
    let res = eval(&cands, seeds, ws);
    let mut best = 0;
    let mut bo = f64::NEG_INFINITY;
    for (i, r) in res.iter().enumerate() {
        let o = tune_obj(r);
        if o > bo {
            bo = o;
            best = i;
        }
    }
    println!("[tune 1-8] {label}: {} configs, best mean quality {:.5} -> {:?}", cands.len(), bo, cands[best]);
    cands[best].clone()
}

fn col<F: Fn(&Out) -> f64>(r: &[Out], f: F) -> Vec<f64> {
    r.iter().map(f).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d5_quality.json".into());
    let (h0, h1) = args
        .iter()
        .position(|a| a == "--held")
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
        .unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let ws = worlds();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    println!("d5_quality: {N_TASKS} tasks/run, {} worlds, tune seeds 1-8, held-out {h0}-{h1}", ws.len());

    let mut eg = vec![];
    for &g in &[1.0, 0.97, 0.9] {
        for &e in &[0.02, 0.05, 0.1] {
            eg.push(Kind::Eps(g, e));
        }
    }
    let eps = tune("eps-greedy", eg, &tune_seeds, &ws);
    let pts = tune("pooled-ts", vec![Kind::Pts(1.0), Kind::Pts(0.99), Kind::Pts(0.97), Kind::Pts(0.9)], &tune_seeds, &ws);
    let mut gg = vec![];
    for &tau in &[0.45, 0.55, 0.65, 0.75] {
        for &z in &[0.5, 1.0] {
            for &disc in &[1.0, 0.97] {
                for &ch in &[false, true] {
                    gg.push(Kind::GuardHebR(GuardConfig { tau, z, kw_a: 2.0, kw_b: 1.0, discount: disc, challenge: ch }));
                }
            }
        }
    }
    let guard_t = tune("guarded(hebr)-retuned", gg, &tune_seeds, &ws);
    let guard_x1 = Kind::GuardHebR(GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true });

    let bnames = ["keyword", "eps-greedy(tuned)", "pooled-ts(tuned)", "hebbian-robust"];
    let bk = vec![Kind::Keyword, eps, pts, Kind::HebR];
    let cnames = ["warm(hebbian-robust)", "guarded(hebr,x1-frozen)", "guarded(hebr,d5-retuned)"];
    let ck = vec![Kind::WarmHebR, guard_x1, guard_t.clone()];
    let all_kinds: Vec<Kind> = bk.iter().chain(ck.iter()).cloned().collect();
    let all_names: Vec<&str> = bnames.iter().chain(cnames.iter()).cloned().collect();
    let res = eval(&all_kinds, &held, &ws);
    let nb = bk.len();

    // ---- tables
    let mut means_json = vec![];
    for (wi, s) in ws.iter().enumerate() {
        println!("\n=== world {} | held-out {h0}-{h1} | mean EXPECTED quality (oracle {:.4}) ===", s.name, summarize(&col(&res[0][wi], |o| o.q[2] + o.regret[2])).mean);
        println!("{:<28}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}{:>9}", "router", "first100", "first250", "all", "last500", "regret", "best%", "real");
        let mut rows = vec![];
        for (ri, n) in all_names.iter().enumerate() {
            let r = &res[ri][wi];
            let m = |f: &dyn Fn(&Out) -> f64| summarize(&col(r, f)).mean;
            println!(
                "{:<28}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>9.4}{:>9.3}{:>9.4}",
                n, m(&|o| o.q[0]), m(&|o| o.q[1]), m(&|o| o.q[2]), m(&|o| o.q[3]), m(&|o| o.regret[2]), m(&|o| o.best_all), m(&|o| o.q_real)
            );
            let dom: Vec<serde_json::Value> = (0..4)
                .map(|d| json!({"domain": DOMS[d], "quality": m(&|o| o.dq[d]), "oracle": m(&|o| o.doracle[d]), "best_frac": m(&|o| o.dbest[d])}))
                .collect();
            rows.push(json!({"router": n, "windows": WINDOWS, "quality": (0..4).map(|k| summarize(&col(r, |o| o.q[k]))).collect::<Vec<_>>(),
                "regret": (0..4).map(|k| m(&|o| o.regret[k])).collect::<Vec<_>>(),
                "best_frac_all": summarize(&col(r, |o| o.best_all)), "best_frac_last500": m(&|o| o.best_last),
                "real_quality": m(&|o| o.q_real), "per_domain": dom}));
        }
        means_json.push(json!({"world": s.name, "rows": rows}));
    }

    // ---- paired tests (overall windows) : candidate vs best fair baseline
    // "fair set" for a candidate = all baselines + the OTHER candidates except the guarded siblings.
    struct T {
        w: usize,
        c: usize,
        win: usize,
        bb: usize,
        d: neural_mesh::stats::PairedDiff,
        dk: f64,
        p: f64,
    }
    let mut tests: Vec<T> = vec![];
    for wi in 0..ws.len() {
        for win in 0..4 {
            for (cj, _) in cnames.iter().enumerate() {
                let ci = nb + cj;
                // baselines for this candidate: the 4 base routers, plus warm (if candidate is guarded)
                let mut pool: Vec<usize> = (0..nb).collect();
                if cj != 0 {
                    pool.push(nb);
                }
                let bb = *pool
                    .iter()
                    .max_by(|&&a, &&b| summarize(&col(&res[a][wi], |o| o.q[win])).mean.partial_cmp(&summarize(&col(&res[b][wi], |o| o.q[win])).mean).unwrap())
                    .unwrap();
                let (c, b) = (col(&res[ci][wi], |o| o.q[win]), col(&res[bb][wi], |o| o.q[win]));
                tests.push(T { w: wi, c: cj, win, bb, d: paired_diff(&c, &b), dk: summarize(&c).mean - summarize(&col(&res[0][wi], |o| o.q[win])).mean, p: paired_t_pvalue(&c, &b) });
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.p).collect::<Vec<_>>());
    println!("\n=== candidate vs BEST fair baseline (paired, by window); Holm over {} tests ===", tests.len());
    let mut tj = vec![];
    for (i, t) in tests.iter().enumerate() {
        let sig = adj[i] < 0.05;
        let v = if sig && t.d.diff.mean > 0.0 { "WIN" } else if sig { "LOSS" } else { "ns" };
        println!(
            "{:<20} {:<26} {:<9} vs {:<20} {:+.4} [{:+.4},{:+.4}] holm {:.4} {:<4} | vs kw {:+.4}",
            ws[t.w].name, cnames[t.c], WINDOWS[t.win], all_names[t.bb], t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, adj[i], v, t.dk
        );
        tj.push(json!({"world": ws[t.w].name, "cand": cnames[t.c], "window": WINDOWS[t.win], "best_baseline": all_names[t.bb],
            "diff": t.d.diff, "p_raw": t.p, "holm_p": adj[i], "verdict": v, "diff_vs_keyword": t.dk}));
    }

    // ---- per-domain tests (overall period), best baseline chosen per domain
    let mut dt: Vec<(usize, usize, usize, usize, neural_mesh::stats::PairedDiff, f64)> = vec![];
    for wi in 0..ws.len() {
        for d in 0..4 {
            for cj in 0..cnames.len() {
                let ci = nb + cj;
                let mut pool: Vec<usize> = (0..nb).collect();
                if cj != 0 {
                    pool.push(nb);
                }
                let bb = *pool
                    .iter()
                    .max_by(|&&a, &&b| summarize(&col(&res[a][wi], |o| o.dq[d])).mean.partial_cmp(&summarize(&col(&res[b][wi], |o| o.dq[d])).mean).unwrap())
                    .unwrap();
                let (c, b) = (col(&res[ci][wi], |o| o.dq[d]), col(&res[bb][wi], |o| o.dq[d]));
                dt.push((wi, d, cj, bb, paired_diff(&c, &b), paired_t_pvalue(&c, &b)));
            }
        }
    }
    let dadj = holm_adjust(&dt.iter().map(|t| t.5).collect::<Vec<_>>());
    println!("\n=== per-domain (all tasks) candidate vs best baseline; Holm over {} ===", dt.len());
    let mut dj = vec![];
    for (i, t) in dt.iter().enumerate() {
        let sig = dadj[i] < 0.05;
        let v = if sig && t.4.diff.mean > 0.0 { "WIN" } else if sig { "LOSS" } else { "ns" };
        println!("{:<20} {:<10} {:<26} vs {:<20} {:+.4} [{:+.4},{:+.4}] holm {:.4} {}", ws[t.0].name, DOMS[t.1], cnames[t.2], all_names[t.3], t.4.diff.mean, t.4.diff.lo, t.4.diff.hi, dadj[i], v);
        dj.push(json!({"world": ws[t.0].name, "domain": DOMS[t.1], "cand": cnames[t.2], "best_baseline": all_names[t.3], "diff": t.4.diff, "holm_p": dadj[i], "verdict": v}));
    }

    // ---- guarded vs warm direct, and best-agent fraction diffs
    let mut gw = vec![];
    println!("\n=== guarded(x1-frozen) - warm, all-tasks quality and best%% ===");
    for wi in 0..ws.len() {
        let d = paired_diff(&col(&res[nb + 1][wi], |o| o.q[2]), &col(&res[nb][wi], |o| o.q[2]));
        let db = paired_diff(&col(&res[nb + 1][wi], |o| o.best_all), &col(&res[nb][wi], |o| o.best_all));
        println!("{:<20} dq {:+.4} [{:+.4},{:+.4}]  dbest {:+.3} [{:+.3},{:+.3}]", ws[wi].name, d.diff.mean, d.diff.lo, d.diff.hi, db.diff.mean, db.diff.lo, db.diff.hi);
        gw.push(json!({"world": ws[wi].name, "dq": d.diff, "dbest": db.diff}));
    }

    let j = json!({
        "held_out": format!("{h0}-{h1}"),
        "protocol": "tune seeds 1-8 (eps, pts, guarded retune; objective = mean quality over 4 windows x worlds); paired t vs best fair baseline per (world,window); Holm over cand x world x window; per-domain separately",
        "world_assumptions": "synthetic: 16 sub-skills in 4 domains, 8 hidden agents; expected quality = clamp(sum_w skill*(1-0.3*difficulty)); realized = + agent noise, or crash->0.05; tasks pre-drawn so routers are paired; debugging tasks carry a language-keyword decoy w.p. .35 in every world",
        "tuned": json!({"eps": format!("{:?}", bk[1]), "pts": format!("{:?}", bk[2]), "guard_retuned": format!("{:?}", guard_t)}),
        "means": means_json, "tests": tj, "domain_tests": dj, "guarded_minus_warm": gw
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
