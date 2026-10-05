//! H11-cost (R5-C TOKEN EFFICIENCY): the cost model itself. Prints the units, the synthetic token/price assumptions, the conversion of encoder milliseconds into
//! token-equivalents (TE), the break-even prices at which each cost component starts to matter, and a measured CPU-time-per-task of the Rust decision loop.
//! SYNTHETIC accounting; not a real token bill.
//!
//! Usage: cargo run --release --offline --bin h11_cost

use std::time::Instant;

#[path = "h11_common/mod.rs"]
mod h11;
use h11::pworld::*;
use h11::*;

fn main() {
    println!("=== H11 cost model (SYNTHETIC token-equivalents, TE) ===");
    println!("unit: 1 TE = one prompt token at the reference (pi = 1, mid-price) generator.  completion token = r_out TE (default 4, swept 1/2/4/8).");
    println!("agent price pi_a multiplies BOTH prompt and completion tokens of that agent's calls. Verifier: its own token counts; v_scale sweeps its price.");
    println!("encoder/retrieval compute: lam_ms TE per ms. Default 0.0046 TE/ms: a $0.05/vCPU-hour host (1.39e-8 $/ms) against a $3/Mtok prompt price (3e-6 $/TE).");
    println!("   ms inputs: encode 2.6 ms/text (python ONNX MiniLM, results/f6_py_latency.json; 2.4 ms in results/h1_embed_ident.json), scan 0.02-0.05 ms,");
    println!("              compute-once memo: encoder calls/decision 3.64 -> 0.98 (cap 4) -> 0.39 (cap 4096) in the repeat-heavy f5/f6 worlds (results/h1_embed_ident.json; not re-measured here).");
    println!("   lam sweep: 0.0046 (cloud CPU) / 0.1 (accelerator-priced) / 1 / 10 (pathological). rerank: kappa TE per candidate pair (0 here: no rerank arm; g16/h2 rejected rerank).");
    println!("   pc_disc: multiplier on the static base prompt (prompt-cache style, 1 = none). ctx_scale: multiplier on injected-context tokens.");
    println!();
    println!("--- world P (g18 copy, multi-step pipeline) token assumptions ---");
    println!("  generator step call: prompt {B_STEP} + {DEP_TOK}/dependency (+ {C_NOTE} per injected note x gamma), completion {O_STEP}  => S_REF = {S_REF} TE at pi = 1, r_out = 4");
    println!("  verifier call: cv x S_REF TE (cv = .3 default; .8 lat-pricever; 1.0 ctl-rare-pricever): {V_OUT} completion tokens + (cv*S_REF - 4*{V_OUT}) prompt tokens");
    println!("  retrieval: {EMB_MS} ms encode + {SEARCH_MS} ms scan per step (compute-once: encode only on the first sight of a (project, step type) key)");
    println!("--- world D (document QA on g16 fixtures) token assumptions ---");
    println!("  generator call: prompt 300 + injected docs (real chars/4 tokens x ctx_scale 4 = ~297 TE per doc), completion 250 => S_REF_D = 1300 TE at pi = 1 (no context)");
    println!("  verifier call: 250 prompt + the same evidence docs + 40 completion; cached-answer reverify costs one verifier call");
    println!("  agent prices (pi): 0.25, 0.25, 1, 1, 3, 3 with caps .50/.50/.68/.68/.86/.86 (synthetic); P prices: Cap = clamp(.25 + 2.5 (mean competence - .4), .25, 3) [also Rand, Flat sensitivity]");
    println!();
    println!("--- worked example: pipe-main, always-verify session L3 (mean per task, seeds 31-40) ---");
    let ws = worlds();
    let seeds: Vec<u64> = (31..=40).collect();
    let mut acc = Res::default();
    let mut q = 0.0;
    let t0 = Instant::now();
    let mut tasks = 0usize;
    for &s in &seeds {
        let inst = build(&ws[0], s, 0);
        let r = run_p(&ws[0], &inst, &PCfg::l3(), s);
        acc.add(&r.res);
        q += r.q;
        tasks += inst.tasks.len();
    }
    let el_us = t0.elapsed().as_secs_f64() * 1e6 / tasks as f64;
    let res = acc.scaled(1.0 / seeds.len() as f64);
    let p = Prices::base();
    let total = cost_te(&res, &p);
    println!("  q {:.3}; per task: {:.2} generator calls, {:.2} verifier calls, {:.2} retrievals ({:.2} injected notes), {:.1} ms encoder", q / seeds.len() as f64, res.n_gen, res.n_ver, res.n_ret, res.n_inj, res.ms);
    for (n, x) in decompose(&res, &p) {
        println!("    {:<22} {:>8.1} TE  {:>5.1}%", n, x, 100.0 * x / total);
    }
    println!("    {:<22} {:>8.1} TE", "TOTAL", total);
    println!("  measured Rust decision-loop CPU (router + gates + bookkeeping): {:.1} us/task (noisy box) = {:.5} TE at lam 0.0046 -> {:.2e} of the task cost", el_us, el_us / 1000.0 * p.lam_ms, el_us / 1000.0 * p.lam_ms / total);
    println!();
    println!("--- break-even prices for the minor components (pipe-main L3) ---");
    let ms_share = |lam: f64| 100.0 * lam * res.ms / cost_te(&res, &Prices { lam_ms: lam, ..p });
    for lam in [0.0046, 0.1, 1.0, 10.0, 100.0] {
        println!("  lam_ms {:>7}: encoder compute = {:.3}% of the task cost; with compute-once {:.3}%", lam, ms_share(lam), 100.0 * lam * res.ms_cached / cost_te(&res, &Prices { lam_ms: lam, embed_once: true, ..p }));
    }
    let lam_1pct = 0.01 * (total - 0.0) / res.ms / (1.0 - 0.01);
    println!("  => encoder compute reaches 1% of the L3 task cost only at lam_ms ~ {:.1} TE/ms (~{:.0}x the cloud-CPU price); at the default it is {:.4}%", lam_1pct, lam_1pct / p.lam_ms, ms_share(p.lam_ms));
    println!("  prompt-cache base x0.1: total {:.0} TE ({:+.1}%); verifier x0.25: {:.0} TE ({:+.1}%); r_out=8: {:.0} TE ({:+.1}%); r_out=1: {:.0} TE ({:+.1}%)", cost_te(&res, &Prices { pc_disc: 0.1, ..p }), 100.0 * (cost_te(&res, &Prices { pc_disc: 0.1, ..p }) / total - 1.0), cost_te(&res, &Prices { v_scale: 0.25, ..p }), 100.0 * (cost_te(&res, &Prices { v_scale: 0.25, ..p }) / total - 1.0), cost_te(&res, &Prices { r_out: 8.0, ..p }), 100.0 * (cost_te(&res, &Prices { r_out: 8.0, ..p }) / total - 1.0), cost_te(&res, &Prices { r_out: 1.0, ..p }), 100.0 * (cost_te(&res, &Prices { r_out: 1.0, ..p }) / total - 1.0));
    println!();
    println!("--- price scenarios used by the sensitivity tables (world P / world D) ---");
    for (n, s) in scenarios(false) {
        println!("  P  {:<34} r_out {} lam_ms {} pc_disc {} ctx_scale {} v_scale {} embed_once {}", n, s.r_out, s.lam_ms, s.pc_disc, s.ctx_scale, s.v_scale, s.embed_once);
    }
    for (n, s) in scenarios(true) {
        println!("  D  {:<34} r_out {} lam_ms {} pc_disc {} ctx_scale {} v_scale {} embed_once {}", n, s.r_out, s.lam_ms, s.pc_disc, s.ctx_scale, s.v_scale, s.embed_once);
    }
}
