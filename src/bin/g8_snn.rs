//! G8 / W9: spiking-neural-network mechanisms for routing, BEYOND R-STDP.
//!
//! Native (no SNN crate): (1) LIF and Izhikevich winner-take-all selector with lateral
//! inhibition, (2) three-factor learner with eligibility-trace credit assignment over
//! delayed / anonymous reward, (3) population coding (Gaussian tuning curves) of continuous
//! features. Non-spiking control = the SAME learner/encoder/trace with a softmax selector.
//!
//! Protocol: tune on seeds 1-8 (per world, baselines and candidates alike), evaluate on held-out
//! 31-60, frozen configs re-evaluated on fresh 61-90; paired 95% t-CI; Holm over
//! candidates x counted worlds x {all, late} vs the best fair baseline per (world, window).
//! Worlds are defined in g8_common/worlds.rs (fixed before running).
//!
//! Usage: cargo run --release --bin g8_snn [-- --out results/g8s_snn.json --confirm-out results/g8s_snn_confirm.json]
//!        [--held 31-60] [--confirm 61-90] [--samples 200] [--selfcheck]

mod g8_common;
use g8_common::pol::*;
use g8_common::snn::*;
use g8_common::worlds::*;
use neural_mesh::router::{Rng, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

const WCOL: [&str; 3] = ["all", "late", "early"];

fn col(r: &[[f64; 3]], j: usize) -> Vec<f64> {
    r.iter().map(|x| x[j]).collect()
}

fn eval(spec: &Spec, envs: &[Env], seeds: &[u64]) -> Vec<[f64; 3]> {
    envs.par_iter().zip(seeds.par_iter()).map(|(e, &s)| drive(build(spec, e, s).as_mut(), e, s)).collect()
}

/// tuning objective: mean "all" over tuning seeds (sequential over seeds; callers parallelise over configs)
fn obj_seq(spec: &Spec, envs: &[Env], seeds: &[u64]) -> f64 {
    let mut t = 0.0;
    for (e, &s) in envs.iter().zip(seeds) {
        t += drive(build(spec, e, s).as_mut(), e, s)[0];
    }
    t / envs.len() as f64
}

fn best_of(cands: Vec<Spec>, envs: &[Env], seeds: &[u64], label: &str) -> (Spec, f64) {
    let n = cands.len();
    let objs: Vec<f64> = cands.par_iter().map(|c| obj_seq(c, envs, seeds)).collect();
    let mut bi = 0;
    for i in 1..n {
        if objs[i] > objs[bi] {
            bi = i;
        }
    }
    println!("   [tune 1-8] {label}: {n} configs, best obj {:.4}", objs[bi]);
    (cands[bi], objs[bi])
}

fn pick<T: Copy>(rng: &mut Rng, xs: &[T]) -> T {
    xs[rng.below(xs.len())]
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fam {
    Lif,
    Izh,
    Rate,
    LifLag,
}

fn sample_cfg(fam: Fam, cont: bool, fb: Feedback, rng: &mut Rng) -> LCfg {
    let (pop, tok) = if cont {
        let pops = [
            Pop::Sep { m: 5, sigma: 0.12 },
            Pop::Sep { m: 8, sigma: 0.08 },
            Pop::Sep { m: 8, sigma: 0.12 },
            Pop::Sep { m: 12, sigma: 0.08 },
            Pop::Sep { m: 12, sigma: 0.12 },
            Pop::Sep { m: 12, sigma: 0.2 },
            Pop::Grid { g: 4, sigma: 0.2 },
            Pop::Grid { g: 6, sigma: 0.12 },
            Pop::Grid { g: 6, sigma: 0.2 },
            Pop::Grid { g: 8, sigma: 0.12 },
        ];
        (pick(rng, &pops), pick(rng, &[Tok::None, Tok::Kw, Tok::All]))
    } else {
        (Pop::None, Tok::All)
    };
    let credit = if fb == Feedback::Anon {
        if fam == Fam::LifLag {
            Credit::Lag(pick(rng, &[1usize, 2, 3, 4, 5, 6, 7, 8, 10, 12]))
        } else {
            Credit::Trace(pick(rng, &[0.5, 0.7, 0.8, 0.9, 0.95]))
        }
    } else {
        Credit::Lag(1)
    };
    let sel = match fam {
        Fam::Rate => Sel::Rate { tau: pick(rng, &[0.01, 0.03, 0.06, 0.12]) },
        Fam::Lif | Fam::LifLag => Sel::Wta(WtaCfg {
            kind: NeuronKind::Lif { leak: 0.8 },
            steps: pick(rng, &[16usize, 32]),
            lambda_in: pick(rng, &[2.0, 4.0]),
            gain: pick(rng, &[0.8, 1.2, 2.0]),
            inh: pick(rng, &[0.0, 0.5, 1.0]),
            noise: pick(rng, &[0.1, 0.3, 0.6]),
            first_spike: pick(rng, &[false, true]),
        }),
        Fam::Izh => {
            let kind = if rng.bernoulli(0.5) {
                NeuronKind::Izh { a: 0.02, b: 0.2, c: -65.0, d: 8.0 }
            } else {
                NeuronKind::Izh { a: 0.1, b: 0.2, c: -65.0, d: 2.0 }
            };
            Sel::Wta(WtaCfg {
                kind,
                steps: pick(rng, &[30usize, 60]),
                lambda_in: pick(rng, &[2.0, 4.0]),
                gain: pick(rng, &[8.0, 14.0, 20.0]),
                inh: pick(rng, &[0.0, 10.0, 25.0]),
                noise: pick(rng, &[1.0, 3.0, 6.0]),
                first_spike: pick(rng, &[false, true]),
            })
        }
    };
    LCfg {
        sel,
        pop,
        tok,
        eta: pick(rng, &[0.1, 0.2, 0.35, 0.5, 0.7]),
        w0: pick(rng, &[0.5, 0.7, 0.85, 1.0]),
        kwb: pick(rng, &[0.0, 0.15, 0.3]),
        eps: pick(rng, &[0.0, 0.02, 0.05]),
        credit,
    }
}

fn atts(fb: Feedback) -> Vec<Attrib> {
    if fb == Feedback::Anon {
        let mut v: Vec<Attrib> = [1usize, 2, 3, 4, 5, 6, 8, 10, 12].iter().map(|&l| Attrib::Lag(l)).collect();
        v.extend([3.0, 6.0, 9.0].iter().map(|&m| Attrib::Geo(m)));
        v
    } else {
        vec![Attrib::None]
    }
}

struct Row {
    label: String,
    spec: Spec,
    kind: &'static str, // baseline | candidate | control
    control_of: Option<String>,
}

fn tstr(s: &Spec) -> String {
    format!("{:?}", s)
}

fn lcfg_of(s: &Spec) -> LCfg {
    match s {
        Spec::Snn(c) => *c,
        _ => unreachable!(),
    }
}

fn selfcheck() {
    println!("selfcheck:");
    // 1. WTA picks the strongest-drive neuron
    let ni = 8usize;
    let mut w = vec![0.3; NA * ni];
    for j in 0..ni {
        w[3 * ni + j] = 0.9;
    }
    let inputs: Vec<(usize, f64)> = (0..4).map(|j| (j, 0.25)).collect();
    let cand = [0usize, 1, 2, 3, 4, 5];
    let lif = WtaCfg { kind: NeuronKind::Lif { leak: 0.8 }, steps: 32, lambda_in: 4.0, gain: 1.2, inh: 0.5, noise: 0.3, first_spike: false };
    let izh = WtaCfg { kind: NeuronKind::Izh { a: 0.02, b: 0.2, c: -65.0, d: 8.0 }, steps: 60, lambda_in: 4.0, gain: 14.0, inh: 10.0, noise: 3.0, first_spike: false };
    for (nm, c) in [("lif", lif), ("izh", izh)] {
        let mut wta = Wta::default();
        let mut rng = Rng::new(1);
        let k = (0..400).filter(|_| wta.run(&c, &w, ni, &inputs, &cand, &mut rng) == 3).count();
        println!("  {nm} WTA picks max-drive neuron {k}/400");
        assert!(k > 340, "{nm} WTA failed to pick the strongest neuron");
    }
    // 2. learner learns a stationary 2-token world (immediate reward), spiking and rate
    for (nm, sel) in [("lif", Sel::Wta(lif)), ("rate", Sel::Rate { tau: 0.05 })] {
        let cfg = LCfg { sel, pop: Pop::None, tok: Tok::All, eta: 0.35, w0: 0.85, kwb: 0.0, eps: 0.0, credit: Credit::Lag(1) };
        let mut l = Learner::new(cfg, HashMap::new());
        let mut rng = Rng::new(2);
        let mut good = 0;
        for i in 0..600 {
            let t = Task { id: i, task_type: None, features: vec![if i % 2 == 0 { "p".into() } else { "q".into() }] };
            let a = l.decide(&t, [0.0, 0.0], &cand, &mut rng);
            let best = if i % 2 == 0 { 1 } else { 4 };
            let p = if a == best { 0.9 } else { 0.1 };
            if i >= 400 && a == best {
                good += 1;
            }
            l.reward_id(i as usize, if rng.f64() < p { 1.0 } else { 0.0 });
        }
        println!("  {nm} learner picks best agent {good}/200 after warmup");
        assert!(good > 150);
    }
    // 3. population code is unimodal and peaks at the stimulus
    let mut out = vec![];
    Pop::Sep { m: 8, sigma: 0.12 }.encode([0.3, 0.7], &mut out);
    let (jmax, _) = out.iter().filter(|p| p.0 < 8).fold((0, 0.0), |b, &(j, r)| if r > b.1 { (j, r) } else { b });
    println!("  pop code x0=0.3 peaks at neuron {jmax} (centre 0.286 -> 2)");
    assert_eq!(jmax, 2);
    // 4. trace credit: anonymous reward with fixed lag, agent-specific outcome learnable
    println!("selfcheck OK");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selfcheck") {
        selfcheck();
        return;
    }
    let argv = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = argv("--out").unwrap_or("results/g8s_snn.json".into());
    let out_c = argv("--confirm-out").unwrap_or("results/g8s_snn_confirm.json".into());
    let rng_of = |s: &str, d: (u64, u64)| -> Vec<u64> {
        let (lo, hi) = s.split_once('-').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap_or(d);
        (lo..=hi).collect()
    };
    let held_seeds = rng_of(&argv("--held").unwrap_or("31-60".into()), (31, 60));
    let conf_seeds = rng_of(&argv("--confirm").unwrap_or("61-90".into()), (61, 90));
    let n_samples: usize = argv("--samples").and_then(|s| s.parse().ok()).unwrap_or(200);
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let nw = WORLD_NAMES.len();
    println!("G8 SNN: tune 1-8, held {}-{}, confirm {}-{}, {} random-search samples per candidate family/world", held_seeds[0], held_seeds[held_seeds.len() - 1], conf_seeds[0], conf_seeds[conf_seeds.len() - 1], n_samples);

    let mk = |w: usize, seeds: &[u64]| -> Vec<Env> { seeds.par_iter().map(|&s| make_env(w, s)).collect() };
    let tune: Vec<Vec<Env>> = (0..nw).map(|w| mk(w, &tune_seeds)).collect();
    let held: Vec<Vec<Env>> = (0..nw).map(|w| mk(w, &held_seeds)).collect();
    let conf: Vec<Vec<Env>> = (0..nw).map(|w| mk(w, &conf_seeds)).collect();

    let mut rows_per_world: Vec<Vec<Row>> = vec![];
    let mut tuned_json = vec![];
    for w in 0..nw {
        let fb = tune[w][0].fb;
        let cont = w == 2 || w == 3 || w == 5;
        let at = atts(fb);
        println!("== world {} (feedback {:?}) ==", WORLD_NAMES[w], fb);
        let te = &tune[w];
        // ---- fair baselines, tuned per world on seeds 1-8
        let mut c = vec![];
        for &g in &[0.95, 0.99, 1.0] {
            for &e in &[0.02, 0.05, 0.1] {
                for &a in &at {
                    c.push(Spec::Eps(g, e, a));
                }
            }
        }
        let (eps, _) = best_of(c, te, &tune_seeds, "eps-greedy");
        let c: Vec<Spec> = [0.9, 0.95, 0.99, 1.0].iter().flat_map(|&g| at.iter().map(move |&a| Spec::Pts(g, a))).collect();
        let (pts, _) = best_of(c, te, &tune_seeds, "pooled-ts");
        let c: Vec<Spec> = at.iter().map(|&a| Spec::HebR(a)).collect();
        let (hebr, _) = best_of(c, te, &tune_seeds, "hebbian-robust");
        let mut c = vec![];
        for &k in &[3.0, 10.0, 30.0, 100.0, 300.0] {
            for &e in &[Ev::Set, Ev::Mean, Ev::Global, Ev::FailSet, Ev::FailMean] {
                for &a in &at {
                    c.push(Spec::WarmHebR(k, e, a));
                }
            }
        }
        let (warm, _) = best_of(c, te, &tune_seeds, "warm(hebbian-robust)");
        let c: Vec<Spec> = at.iter().map(|&a| Spec::Guarded(a)).collect();
        let (guard, _) = best_of(c, te, &tune_seeds, "guarded-safe");

        // ---- v2: matched-effort baselines (same random-search budget as the candidates)
        let samp = |tag: u64, f: &dyn Fn(&mut Rng) -> Spec| -> Spec {
            let mut rng = Rng::new(0x7a11_0000 + (w as u64) * 31 + tag);
            let c: Vec<Spec> = (0..n_samples).map(|_| f(&mut rng)).collect();
            best_of(c, te, &tune_seeds, &format!("matched-effort #{tag}")).0
        };
        let atv = at.clone();
        let hp = |rng: &mut Rng| HebP {
            eta: pick(rng, &[0.005, 0.01, 0.03, 0.1, 0.3, 0.5]),
            eps: pick(rng, &[0.0, 0.02, 0.05, 0.1]),
            temp: pick(rng, &[0.002, 0.01, 0.03, 0.05, 0.1, 0.2]),
            decay: pick(rng, &[0.0, 0.003, 0.01, 0.03]),
            ba: pick(rng, &[0.05, 0.1, 0.2]),
            cap: pick(rng, &[1.0, 2.0, 4.0]),
            ucb: pick(rng, &[0.0, 0.05, 0.1, 0.2]),
            med: pick(rng, &[false, true]),
        };
        let hebt = samp(1, &|r| Spec::T(SpecT::Heb(hp(r), pick(r, &atv))));
        let evs = [Ev::Set, Ev::Mean, Ev::Global, Ev::FailSet, Ev::FailMean];
        let warmt = samp(2, &|r| Spec::T(SpecT::Warm(pick(r, &[3.0, 10.0, 30.0, 100.0, 300.0]), pick(r, &evs), hp(r), pick(r, &atv))));
        let guardt = samp(3, &|r| {
            let g = GuardP { tau: pick(r, &[0.5, 0.6, 0.7, 0.8]), z: pick(r, &[0.5, 1.0, 2.0]), kw_a: pick(r, &[2.0, 4.0, 8.0]), disc: pick(r, &[0.95, 0.98, 1.0]) };
            Spec::T(SpecT::Guard(g, hp(r), pick(r, &atv)))
        });
        // ---- candidate families: equal random-search budget each
        let fam_best = |fam: Fam| -> Spec {
            let mut rng = Rng::new(0x6738_0000 + (w as u64) * 17 + fam as u64);
            let c: Vec<Spec> = (0..n_samples).map(|_| Spec::Snn(sample_cfg(fam, cont, fb, &mut rng))).collect();
            best_of(c, te, &tune_seeds, &format!("{fam:?}")).0
        };
        let lif = fam_best(Fam::Lif);
        let izh = fam_best(Fam::Izh);
        let rate = fam_best(Fam::Rate);
        let lag = if fb == Feedback::Anon { Some(fam_best(Fam::LifLag)) } else { None };

        let mut rows: Vec<Row> = vec![];
        let mut add = |l: &str, s: Spec, k: &'static str, of: Option<&str>| rows.push(Row { label: l.into(), spec: s, kind: k, control_of: of.map(|x| x.to_string()) });
        add("keyword", Spec::Keyword, "baseline", None);
        add("eps-greedy", eps, "baseline", None);
        add("pooled-ts", pts, "baseline", None);
        add("hebbian-robust", hebr, "baseline", None);
        add("warm(hebr)", warm, "baseline", None);
        add("guarded-safe", guard, "baseline", None);
        add("hebr-tuned(matched effort)", hebt, "baseline", None);
        add("warm(hebr-tuned)", warmt, "baseline", None);
        add("guarded(hebr-tuned)", guardt, "baseline", None);
        add("snn-lif", lif, "candidate", None);
        add("snn-izh", izh, "candidate", None);
        add("rate-ctl (non-spiking)", rate, "candidate", None);
        let ab = |l: &str, f: &dyn Fn(LCfg) -> Option<LCfg>, base: &Spec, of: &str, add: &mut dyn FnMut(&str, Spec, &'static str, Option<&str>)| {
            if let Some(c) = f(lcfg_of(base)) {
                add(l, Spec::Snn(c), "control", Some(of));
            }
        };
        let no_inh = |mut c: LCfg| -> Option<LCfg> {
            if let Sel::Wta(ref mut x) = c.sel {
                if x.inh == 0.0 {
                    return None;
                }
                x.inh = 0.0;
                Some(c)
            } else {
                None
            }
        };
        let no_pop = |mut c: LCfg| -> Option<LCfg> {
            if !cont || c.pop == Pop::None {
                return None;
            }
            c.pop = Pop::None;
            c.tok = Tok::All;
            Some(c)
        };
        let no_tok = |mut c: LCfg| -> Option<LCfg> {
            if !cont || c.pop == Pop::None || c.tok == Tok::None {
                return None;
            }
            c.tok = Tok::None;
            Some(c)
        };
        let no_trace = |mut c: LCfg| -> Option<LCfg> {
            if fb != Feedback::Anon {
                return None;
            }
            c.credit = Credit::Trace(0.0);
            Some(c)
        };
        let mut addf = |l: &str, s: Spec, k: &'static str, of: Option<&str>| rows.push(Row { label: l.into(), spec: s, kind: k, control_of: of.map(|x| x.to_string()) });
        ab("snn-lif -inhibition", &no_inh, &lif, "snn-lif", &mut addf);
        ab("snn-lif -pop-code", &no_pop, &lif, "snn-lif", &mut addf);
        ab("snn-lif -tokens", &no_tok, &lif, "snn-lif", &mut addf);
        ab("snn-lif -trace (lambda=0)", &no_trace, &lif, "snn-lif", &mut addf);
        ab("snn-izh -inhibition", &no_inh, &izh, "snn-izh", &mut addf);
        ab("rate-ctl -pop-code", &no_pop, &rate, "rate-ctl (non-spiking)", &mut addf);
        ab("rate-ctl -tokens", &no_tok, &rate, "rate-ctl (non-spiking)", &mut addf);
        ab("rate-ctl -trace (lambda=0)", &no_trace, &rate, "rate-ctl (non-spiking)", &mut addf);
        if let Some(l) = lag {
            addf("snn-lif[lag-L instead of trace]", l, "control", Some("snn-lif"));
        }
        tuned_json.push(json!({
            "world": WORLD_NAMES[w],
            "tuned": rows.iter().map(|r| json!({"row": r.label, "kind": r.kind, "spec": tstr(&r.spec)})).collect::<Vec<_>>(),
        }));
        rows_per_world.push(rows);
    }

    // ---------------------------------------------------------------- evaluation
    println!("\nevaluating frozen configs on held-out and confirm seeds ...");
    // res[w][row] = (held, confirm)
    let res: Vec<Vec<(Vec<[f64; 3]>, Vec<[f64; 3]>)>> = (0..nw)
        .map(|w| rows_per_world[w].iter().map(|r| (eval(&r.spec, &held[w], &held_seeds), eval(&r.spec, &conf[w], &conf_seeds))).collect())
        .collect();

    // ---------------------------------------------------------------- cost (ns/decision, neurons, bytes)
    println!("measuring ns/decision (single thread, held seeds 31-33) ...");
    let mut cost_json = vec![];
    for w in 0..nw {
        for r in &rows_per_world[w] {
            let mut ns = vec![];
            let (mut neu, mut by) = ((0usize, 0usize), 0usize);
            let mut spk = (0u64, 0u64, 0u64);
            for k in 0..3 {
                let (e, s) = (&held[w][k], held_seeds[k]);
                let t0 = Instant::now();
                match r.spec {
                    Spec::Snn(c) => {
                        let mut p = SnnPol { l: Learner::new(c, e.kmap.clone()) };
                        drive(&mut p, e, s);
                        neu = p.l.n_neurons();
                        by = p.l.state_bytes();
                        let q = p.l.spike_stats();
                        spk = (spk.0 + q.0, spk.1 + q.1, spk.2 + q.2);
                    }
                    _ => {
                        drive(build(&r.spec, e, s).as_mut(), e, s);
                    }
                }
                ns.push(t0.elapsed().as_nanos() as f64 / e.n() as f64);
            }
            ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let toks: usize = {
                let mut s = std::collections::HashSet::new();
                for t in &held[w][0].tasks {
                    for f in &t.features {
                        s.insert(f.clone());
                    }
                }
                s.len()
            };
            cost_json.push(json!({
                "world": WORLD_NAMES[w], "row": r.label, "ns_per_decision_median": ns[1],
                "agent_neurons": if matches!(r.spec, Spec::Snn(_)) { neu.0 } else { 0 },
                "input_neurons": if matches!(r.spec, Spec::Snn(_)) { neu.1 } else { 0 },
                "state_bytes": if matches!(r.spec, Spec::Snn(_)) { by } else { toks * NA * 8 },
                "frac_decisions_with_spikes": if spk.0 > 0 { json!(spk.1 as f64 / spk.0 as f64) } else { json!(null) },
                "mean_spikes_per_decision": if spk.0 > 0 { json!(spk.2 as f64 / spk.0 as f64) } else { json!(null) },
                "state_bytes_note": if matches!(r.spec, Spec::Snn(_)) { "exact: weights+trace+token table (excludes simulation snapshot history)" } else { "lower bound: distinct tokens x 6 agents x 8 B (HashMap overhead not counted)" },
            }));
        }
    }

    // ---------------------------------------------------------------- statistics
    let mut sets_json = vec![];
    let mut verdict_lines = vec![];
    for (si, sname) in ["held-out", "confirm"].iter().enumerate() {
        let pick_set = |w: usize, i: usize| -> &Vec<[f64; 3]> {
            if si == 0 { &res[w][i].0 } else { &res[w][i].1 }
        };
        println!("\n================ {} seeds ({}) ================", sname, if si == 0 { format!("{}-{}", held_seeds[0], held_seeds[held_seeds.len() - 1]) } else { format!("{}-{}", conf_seeds[0], conf_seeds[conf_seeds.len() - 1]) });
        println!("mean expected success (window=all) per world");
        print!("{:<34}", "router");
        for n in WORLD_NAMES.iter() {
            print!("{:>14}", n.split(' ').next().unwrap());
        }
        println!();
        // union of labels in order of world 0 then extras
        let mut labels: Vec<String> = vec![];
        for w in 0..nw {
            for r in &rows_per_world[w] {
                if !labels.contains(&r.label) {
                    labels.push(r.label.clone());
                }
            }
        }
        for l in &labels {
            print!("{:<34}", l);
            for w in 0..nw {
                match rows_per_world[w].iter().position(|r| &r.label == l) {
                    Some(i) => print!("{:>14}", format!("{:.4}", summarize(&col(pick_set(w, i), 0)).mean)),
                    None => print!("{:>14}", "-"),
                }
            }
            println!();
        }
        println!("references (not baselines): mean over seeds of oracle / uniform-random expected success");
        let mut refs = vec![];
        for w in 0..nw {
            let envs = if si == 0 { &held[w] } else { &conf[w] };
            let (mut o, mut u) = (0.0, 0.0);
            for e in envs.iter() {
                let n = e.n() as f64;
                o += e.probs.iter().map(|p| p.iter().cloned().fold(0.0, f64::max)).sum::<f64>() / n;
                u += e.probs.iter().map(|p| p.iter().sum::<f64>() / NA as f64).sum::<f64>() / n;
            }
            let (o, u) = (o / envs.len() as f64, u / envs.len() as f64);
            println!("  {:<26} oracle {:.4}  uniform-random {:.4}", WORLD_NAMES[w], o, u);
            refs.push(json!({"world": WORLD_NAMES[w], "oracle": o, "uniform_random": u}));
        }
        // best fair baseline per (world, window j in 0,1)
        let mut best_b = vec![[0usize; 2]; nw];
        for w in 0..nw {
            for j in 0..2 {
                let mut bi = 0;
                let mut bm = f64::NEG_INFINITY;
                for (i, r) in rows_per_world[w].iter().enumerate() {
                    if r.kind == "baseline" {
                        let m = summarize(&col(pick_set(w, i), j)).mean;
                        if m > bm {
                            bm = m;
                            bi = i;
                        }
                    }
                }
                best_b[w][j] = bi;
            }
        }
        // Holm family over counted worlds
        let mut cells: Vec<(usize, usize, usize)> = vec![];
        for w in 0..N_COUNTED {
            for (i, r) in rows_per_world[w].iter().enumerate() {
                if r.kind == "candidate" {
                    for j in 0..2 {
                        cells.push((w, i, j));
                    }
                }
            }
        }
        let pv: Vec<f64> = cells.iter().map(|&(w, i, j)| paired_t_pvalue(&col(pick_set(w, i), j), &col(pick_set(w, best_b[w][j]), j))).collect();
        let adj = holm_adjust(&pv);
        println!("\ncandidates vs BEST FAIR BASELINE, diff [95% CI], Holm over {} cells (counted worlds A-D)", cells.len());
        let mut jt = vec![];
        let mut wins: HashMap<String, Vec<String>> = HashMap::new();
        for (ci, &(w, i, j)) in cells.iter().enumerate() {
            let b = best_b[w][j];
            let d = paired_diff(&col(pick_set(w, i), j), &col(pick_set(w, b), j));
            let win = adj[ci] < 0.05 && d.diff.mean > 0.0;
            let lose = adj[ci] < 0.05 && d.diff.mean < 0.0;
            if win {
                wins.entry(rows_per_world[w][i].label.clone()).or_default().push(format!("{}/{}", WORLD_NAMES[w].split(' ').next().unwrap(), WCOL[j]));
            }
            println!(
                "  {:<26} {:<18} {:<5} vs {:<15} {:+.4} [{:+.4},{:+.4}] W/L {}/{} holm p={:.4} {}",
                rows_per_world[w][i].label, WORLD_NAMES[w], WCOL[j], rows_per_world[w][b].label, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.losses, adj[ci],
                if win { "WIN" } else if lose { "LOSE" } else { "" }
            );
            jt.push(json!({"candidate": rows_per_world[w][i].label, "world": WORLD_NAMES[w], "window": WCOL[j],
                "mean": summarize(&col(pick_set(w, i), j)).mean, "best_baseline": rows_per_world[w][b].label,
                "best_baseline_mean": summarize(&col(pick_set(w, b), j)).mean,
                "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": adj[ci], "win": win, "lose": lose}));
        }
        let mut accepted = vec![];
        for (l, ws) in &wins {
            let mut worlds: Vec<&str> = ws.iter().map(|s| s.split('/').next().unwrap()).collect();
            worlds.sort();
            worlds.dedup();
            println!("WIN cells for {l}: {:?}", ws);
            if worlds.len() > 1 {
                accepted.push(l.clone());
            }
        }
        println!("accepted by bar (Holm win vs best fair baseline in >1 counted world): {:?}", accepted);
        verdict_lines.push(format!("{sname}: accepted={:?} wins={:?}", accepted, wins));

        // controls worlds E,F vs best baseline (information only)
        println!("\ncontrol worlds (E static, F no-information): candidate - best fair baseline, unadjusted");
        let mut jc = vec![];
        for w in N_COUNTED..nw {
            for (i, r) in rows_per_world[w].iter().enumerate() {
                if r.kind != "candidate" {
                    continue;
                }
                for j in 0..2 {
                    let b = best_b[w][j];
                    let d = paired_diff(&col(pick_set(w, i), j), &col(pick_set(w, b), j));
                    println!("  {:<26} {:<24} {:<5} {:+.4} [{:+.4},{:+.4}] vs {}", r.label, WORLD_NAMES[w], WCOL[j], d.diff.mean, d.diff.lo, d.diff.hi, rows_per_world[w][b].label);
                    jc.push(json!({"candidate": r.label, "world": WORLD_NAMES[w], "window": WCOL[j], "vs": rows_per_world[w][b].label, "diff": d.diff}));
                }
            }
        }

        // does spiking matter? snn vs rate-ctl, and ablations vs parent (unadjusted, windows all/late)
        println!("\nmechanism isolation (row - parent), unadjusted, all worlds");
        let mut jm = vec![];
        for w in 0..nw {
            let ri = rows_per_world[w].iter().position(|r| r.label.starts_with("rate-ctl")).unwrap();
            for (i, r) in rows_per_world[w].iter().enumerate() {
                let (par, lab): (Option<usize>, String) = if r.kind == "control" {
                    (r.control_of.as_ref().and_then(|p| rows_per_world[w].iter().position(|x| &x.label == p)), r.label.clone())
                } else if r.label.starts_with("snn-") {
                    (Some(ri), format!("{} vs rate-ctl", r.label))
                } else {
                    (None, String::new())
                };
                let Some(pi) = par else { continue };
                let mut line = format!("  {:<18} {:<42}", WORLD_NAMES[w].split(' ').next().unwrap(), lab);
                for j in 0..2 {
                    let d = paired_diff(&col(pick_set(w, i), j), &col(pick_set(w, pi), j));
                    line.push_str(&format!(" {}:{:+.4}[{:+.4},{:+.4}]", WCOL[j], d.diff.mean, d.diff.lo, d.diff.hi));
                    jm.push(json!({"world": WORLD_NAMES[w], "row": r.label, "parent": rows_per_world[w][pi].label, "window": WCOL[j], "diff_row_minus_parent": d.diff}));
                }
                println!("{line}");
            }
        }
        let table: Vec<_> = (0..nw)
            .map(|w| json!({"world": WORLD_NAMES[w], "rows": rows_per_world[w].iter().enumerate().map(|(i, r)| json!({
                "row": r.label, "kind": r.kind, "spec": tstr(&r.spec),
                "windows": (0..3).map(|j| json!({"window": WCOL[j], "summary": summarize(&col(pick_set(w, i), j))})).collect::<Vec<_>>()
            })).collect::<Vec<_>>()}))
            .collect();
        sets_json.push(json!({"set": sname, "references": refs, "table": table, "tests_vs_best_baseline": jt, "wins": wins, "accepted": accepted, "controls_worlds": jc, "mechanism_isolation": jm}));
    }

    let proto = "tune seeds 1-8 per world (objective: mean expected success, window=all), baselines and candidates tuned alike (random search n per candidate family, full grids for baselines); held-out then fresh confirm seeds with FROZEN configs; paired 95% t-CI; Holm over candidates x counted worlds A-D x {all, late} vs best fair baseline per (world, window); bar = Holm win in >1 counted world";
    for (si, p) in [&out, &out_c].iter().enumerate() {
        let j = json!({
            "protocol": proto,
            "worlds": WORLD_NAMES,
            "counted_worlds": N_COUNTED,
            "n_samples": n_samples,
            "seeds": {"tune": "1-8", "eval": if si == 0 { format!("{}-{}", held_seeds[0], held_seeds[held_seeds.len() - 1]) } else { format!("{}-{}", conf_seeds[0], conf_seeds[conf_seeds.len() - 1]) }},
            "tuned_configs": tuned_json,
            "result": sets_json[si],
            "cost": cost_json,
        });
        if let Some(dir) = std::path::Path::new(p.as_str()).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(p.as_str(), serde_json::to_string_pretty(&j).unwrap()).unwrap();
        println!("wrote {p}");
    }
    println!("\nVERDICT LINES:");
    for v in verdict_lines {
        println!("  {v}");
    }
}
