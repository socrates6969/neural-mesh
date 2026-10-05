//! H5 / R4-B2 (MEMORY-REDUCTION) lever (a), SHARED form: many brains over ONE bounded `SharedVocab`.
//! K = 240 brains (world seeds 31-270) are stepped round-robin (one task each per round) over a single
//! shared vocabulary whose capacity is a fraction of its natural size; evictions by one brain make
//! the others' rows for that string stale (they are swept on the next update). Quality of every brain is
//! compared (paired by seed, TOST +-0.005, 95% CI + Holm over the worlds) with the f64 recommended router
//! run alone on the same world seed. All brains of a world share ONE keyword table (the one of seed 31, `Env::with_kw`;
//! `fork_blank` shares it by design), so the keyword prior is NOT the world-specific one of the other bins: for seeds other than 31
//! it is a (mostly wrong) global prior that the guard has to learn to abandon. Reference and subject see the same table.
//!
//! PRE-DECLARED: worlds big-32agents, big-32agents-longtail, big-32agents-open (the worlds with an evictable
//! tail are longtail and open; big-32agents is the dense control); fractions f of the evictable part of the
//! natural shared vocabulary {1.0 (unbounded control), .5, .25, .125, then .0625 and 0 added after the first run to find the break point}: cap = pinned + f*(natural - pinned);
//! natural = size of the unbounded shared vocabulary after the round-robin run of the same 60 brains.
//! Pass = Holm p<.05 and 95% CI inside +-0.005 in every world. Output: results/h5_shared.{json,log}.

#[path = "h5_common/mod.rs"]
mod common;

use std::time::Instant;

use common::*;
use neural_mesh::compact::{CompactConfig, CompactRouterG, Fx16, Q16, SharedVocab};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::{Rng, Router};
use neural_mesh::stats::holm_adjust;
use serde_json::json;

type U16 = CompactRouterG<Fx16, Q16>;

struct RunOut {
    full: Vec<f64>,
    vocab_len: usize,
    evictions: u64,
    vocab_bytes: usize,
    owned_mean: f64,
    rows_mean: f64,
}

fn run_shared(spec: &WorldSpec, seed_list: &[u64], cap: usize) -> RunOut {
    let kw = spec.make(31).kw_map(); // ONE keyword table for every brain (fork_blank shares it)
    let envs: Vec<Env> = seed_list.iter().map(|&s| spec.make(s).with_kw(kw.clone())).collect();
    let n = envs[0].n_agents();
    let sv = if cap == 0 { SharedVocab::new() } else { SharedVocab::bounded(cap) };
    let proto = U16::with_shared_vocab(n, &kw, CompactConfig::default(), &sv).unwrap();
    let mut brains: Vec<U16> = envs.iter().map(|_| proto.fork_blank()).collect();
    let mut rngs: Vec<Rng> = seed_list.iter().map(|&s| Rng::new(s.wrapping_mul(1_000_003).wrapping_add(17))).collect();
    let mut cum = vec![0.0f64; envs.len()];
    let cands: Vec<usize> = (0..n).collect();
    for t in 0..envs[0].len() {
        for k in 0..envs.len() {
            let task = envs[k].task(t);
            let a = brains[k].select(task, &cands, &mut rngs[k]);
            cum[k] += envs[k].prob(t, a);
            brains[k].update(task, a, envs[k].outcome(t, a));
        }
    }
    let len = envs[0].len() as f64;
    for b in brains.iter_mut() {
        b.shrink_to_fit();
    }
    let owned_mean = brains.iter().map(|b| b.bytes().owned() as f64).sum::<f64>() / brains.len() as f64;
    let rows_mean = brains.iter().map(|b| b.n_rows() as f64).sum::<f64>() / brains.len() as f64;
    RunOut { full: cum.iter().map(|c| c / len).collect(), vocab_len: sv.len(), evictions: sv.evictions(), vocab_bytes: sv.bytes_estimate(), owned_mean, rows_mean }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h5_shared.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    let seed_list = if quick { seeds(31, 38) } else { seeds(31, 270) };
    let worlds = [WorldSpec::Big(32), WorldSpec::BigTail(32), WorldSpec::BigOpen(32)];
    let fracs = [1.0f64, 0.5, 0.25, 0.125, 0.0625, 0.0];
    let mut res_json = Vec::new();
    // reference per world
    let mut table = Vec::new();
    for w in &worlds {
        let refs: Vec<f64> = par_map(&seed_list, |&s| {
            let env = w.make(s).with_kw(w.make(31).kw_map());
            run_policy(&env, &mut recommended_router(env.kw_map(), env.n_agents()), s).full
        });
        let pinned = w.make(31).kw_map().len() as f64;
        let unb = run_shared(w, &seed_list, 0);
        let natural = unb.vocab_len as f64;
        println!("== {} (K={} brains over one vocab): unbounded natural shared vocab {} strings (~{} B est), mean owned/brain {:.0} B, rows/brain {:.1}", w.label(), seed_list.len(), unb.vocab_len, unb.vocab_bytes, unb.owned_mean, unb.rows_mean);
        let mut per_f = Vec::new();
        for &f in &fracs {
            let r = if f >= 1.0 {
                RunOut { full: unb.full.clone(), vocab_len: unb.vocab_len, evictions: 0, vocab_bytes: unb.vocab_bytes, owned_mean: unb.owned_mean, rows_mean: unb.rows_mean }
            } else {
                let cap = ((pinned + f * (natural - pinned).max(0.0)).ceil() as usize).max(1);
                run_shared(w, &seed_list, cap)
            };
            per_f.push((f, r));
        }
        table.push((refs, per_f, natural, pinned));
    }
    // TOST + Holm over worlds per fraction
    for (fi, &f) in fracs.iter().enumerate() {
        let eqs: Vec<EqRes> = (0..worlds.len()).map(|wi| eq_test(&table[wi].1[fi].1.full, &table[wi].0, 0.005)).collect();
        let holm = holm_adjust(&eqs.iter().map(|e| e.p_tost).collect::<Vec<_>>());
        let mut all95 = true;
        for (wi, e) in eqs.iter().enumerate() {
            let c95 = holm[wi] < 0.05 && e.ci95_inside();
            all95 &= c95;
            let r = &table[wi].1[fi].1;
            let verdict = if c95 { "CERTIFIED95" } else if holm[wi] < 0.05 && e.ci90_inside() { "CERTIFIED90" } else if e.lo90 > e.margin || e.hi90 < -e.margin { "NOT equivalent" } else { "inconclusive" };
            println!(
                "  f={f:5} {:24} cap {:>5} vocab {:>5} ids ({:>7} B est)  evictions {:>4}  owned/brain {:>7.0} B rows/brain {:>6.1}   diff {:+.4} 95%CI [{:+.4},{:+.4}] holm p {:.4} -> {verdict}",
                worlds[wi].label(), if f >= 1.0 { "inf".to_string() } else { ((table[wi].3 + f * (table[wi].2 - table[wi].3)).ceil() as usize).to_string() }, r.vocab_len, r.vocab_bytes, r.evictions, r.owned_mean, r.rows_mean, e.mean, e.lo95, e.hi95, holm[wi]
            );
            res_json.push(json!({"world": worlds[wi].label(), "fraction": f, "vocab_ids": r.vocab_len, "vocab_bytes_est": r.vocab_bytes, "evictions": r.evictions, "owned_bytes_per_brain": r.owned_mean, "rows_per_brain": r.rows_mean,
                "diff_mean": e.mean, "ci95": [e.lo95, e.hi95], "ci90": [e.lo90, e.hi90], "holm_p_tost": holm[wi], "p_ni": e.p_ni, "verdict": verdict, "n": e.n}));
        }
        println!("  => f={f}: PASS95 in all worlds: {all95}");
    }
    let j = json!({"bin": "h5_shared", "k_brains": seed_list.len(), "margin": 0.005, "results": res_json, "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
}
