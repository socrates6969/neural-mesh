//! h11 common (R5-C, TOKEN EFFICIENCY): synthetic cost accounting + paired statistics shared by h11_pipe / h11_doc / h11_cost.
//!
//! LABEL: all costs here are SYNTHETIC accounting units ("token-equivalents", TE) computed from per-task RESOURCE COUNTS that the
//! simulated session loops produce. They are not real token bills. Prices are parameters; the engines store resource counts so any price
//! vector can be applied after the fact (see `Prices`, `scenarios()`).
//!
//! UNITS
//!   1 TE  = the price of one prompt (input) token of the reference (mid-price, pi = 1) generator.
//!   completion token = `r_out` TE (default 4: output tokens are typically priced a few x input tokens; swept 1..8).
//!   encoder/retrieval compute: `lam_ms` TE per millisecond. Default 0.0046 TE/ms = a $0.05/vCPU-hour host priced against $3/Mtok input
//!   (0.05/3600 $/s = 1.39e-8 $/ms; /3e-6 $/TE = 0.0046). Swept 0.0046 / 0.1 / 1 / 10 (self-hosted GPU, pessimistic).
//!   ms per encode: 2.6 ms (py ONNX MiniLM, results/f6_py_latency.json) ; 2.4 ms/text in results/h1_embed_ident.json ; scan 0.02 ms.
//!   agent price pi_a multiplies BOTH the prompt and completion tokens of that agent's calls (verifier has its own price `v_scale`).
//!   `pc_disc` = multiplier on the STATIC base prompt (prompt-caching style discount; 1 = none; 0.1 = 90% off cached prefix).
//!   `ctx_scale` = multiplier on injected-context tokens (document-length assumption; stored tokens are the raw model tokens).
#![allow(dead_code)]

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, t_two_sided_p};
use serde_json::{json, Value};

pub mod pworld;

pub fn parse_range(args: &[String], key: &str, default: (u64, u64)) -> Vec<u64> {
    let (a, b) = match args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("range a-b");
            (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap())
        }
        None => default,
    };
    (a..=b).collect()
}
pub fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

// ------------------------------------------------------------------------------------------------ resources
pub const NRES: usize = 16;
pub const RES_NAMES: [&str; NRES] = ["g_base", "g_ctx", "g_out", "v_in", "v_out", "ms", "ms_cached", "rerank", "tools", "n_gen", "n_ver", "n_ret", "n_inj", "n_hit", "unused14", "unused15"];

/// per-task (or per-step-normalised) resource vector. Token fields are already weighted by the AGENT price pi_a.
#[derive(Clone, Copy, Default, Debug)]
pub struct Res {
    pub g_base: f64,
    pub g_ctx: f64,
    pub g_out: f64,
    pub v_in: f64,
    pub v_out: f64,
    pub ms: f64,
    pub ms_cached: f64,
    pub rerank: f64,
    pub tools: f64,
    pub n_gen: f64,
    pub n_ver: f64,
    pub n_ret: f64,
    pub n_inj: f64,
    pub n_hit: f64,
}
impl Res {
    pub fn arr(&self) -> [f64; NRES] {
        [self.g_base, self.g_ctx, self.g_out, self.v_in, self.v_out, self.ms, self.ms_cached, self.rerank, self.tools, self.n_gen, self.n_ver, self.n_ret, self.n_inj, self.n_hit, 0.0, 0.0]
    }
    pub fn from_arr(a: &[f64]) -> Res {
        Res { g_base: a[0], g_ctx: a[1], g_out: a[2], v_in: a[3], v_out: a[4], ms: a[5], ms_cached: a[6], rerank: a[7], tools: a[8], n_gen: a[9], n_ver: a[10], n_ret: a[11], n_inj: a[12], n_hit: a[13] }
    }
    pub fn add(&mut self, o: &Res) {
        let (mut a, b) = (self.arr(), o.arr());
        for i in 0..NRES {
            a[i] += b[i];
        }
        *self = Res::from_arr(&a);
    }
    pub fn scaled(&self, f: f64) -> Res {
        let mut a = self.arr();
        a.iter_mut().for_each(|x| *x *= f);
        Res::from_arr(&a)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Prices {
    pub r_out: f64,
    pub lam_ms: f64,
    pub pc_disc: f64,
    pub ctx_scale: f64,
    pub v_scale: f64,
    pub rr_te: f64,
    pub tool_te: f64,
    pub embed_once: bool,
}
impl Prices {
    pub fn base() -> Prices {
        Prices { r_out: 4.0, lam_ms: 0.0046, pc_disc: 1.0, ctx_scale: 1.0, v_scale: 1.0, rr_te: 0.0, tool_te: 0.0, embed_once: false }
    }
    /// world D: fixture docs are 297 characters (~75 tokens); a realistic RAG chunk is ~4x longer, so the DEFAULT scales context x4 (disclosed; swept 1x..8x)
    pub fn base_d() -> Prices {
        Prices { ctx_scale: 4.0, ..Prices::base() }
    }
}
/// cost in TE of a resource vector
pub fn cost_te(r: &Res, p: &Prices) -> f64 {
    r.g_base * p.pc_disc + r.g_ctx * p.ctx_scale + p.r_out * r.g_out + p.v_scale * (r.v_in + p.r_out * r.v_out) + p.lam_ms * (if p.embed_once { r.ms_cached } else { r.ms }) + p.rr_te * r.rerank + p.tool_te * r.tools
}
/// named decomposition of the cost (TE) for reports
pub fn decompose(r: &Res, p: &Prices) -> [(&'static str, f64); 6] {
    [
        ("gen base prompt", r.g_base * p.pc_disc),
        ("gen injected ctx", r.g_ctx * p.ctx_scale),
        ("gen completion", p.r_out * r.g_out),
        ("verifier", p.v_scale * (r.v_in + p.r_out * r.v_out)),
        ("retrieval/encoder ms", p.lam_ms * (if p.embed_once { r.ms_cached } else { r.ms })),
        ("rerank+tools", p.rr_te * r.rerank + p.tool_te * r.tools),
    ]
}

/// price scenarios for the sensitivity tables (all synthetic; the first one is the default)
pub fn scenarios(world_d: bool) -> Vec<(&'static str, Prices)> {
    let b = if world_d { Prices::base_d() } else { Prices::base() };
    vec![
        ("default", b),
        ("r_out=1 (flat in/out)", Prices { r_out: 1.0, ..b }),
        ("r_out=2", Prices { r_out: 2.0, ..b }),
        ("r_out=8", Prices { r_out: 8.0, ..b }),
        ("lam_ms=0.1 (GPU-priced encoder)", Prices { lam_ms: 0.1, ..b }),
        ("lam_ms=1", Prices { lam_ms: 1.0, ..b }),
        ("lam_ms=10 (pathological)", Prices { lam_ms: 10.0, ..b }),
        ("prompt-cache base x0.1", Prices { pc_disc: 0.1, ..b }),
        ("ctx x1 (short docs/notes)", Prices { ctx_scale: 1.0, ..b }),
        (if world_d { "ctx x8 (long docs)" } else { "ctx x4 (longer notes)" }, Prices { ctx_scale: if world_d { 8.0 } else { 4.0 }, ..b }),
        ("verifier x0.25 (cheap model)", Prices { v_scale: 0.25, ..b }),
        ("verifier x2", Prices { v_scale: 2.0, ..b }),
        ("embed compute-once", Prices { embed_once: true, ..b }),
    ]
}

// ------------------------------------------------------------------------------------------------ stats
#[derive(Clone, Debug)]
pub struct Cmp {
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub p: f64,
    pub tost_p: f64,
    pub n: usize,
}
pub fn one_sided_p(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t, df);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}
/// paired a-b with 95% t-CI, two-sided p, and TOST p for |diff| < eq (eq=0: NaN)
pub fn cmp(a: &[f64], b: &[f64], eq: f64) -> Cmp {
    let d = paired_diff(a, b);
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = diffs.len();
    let s = summarize(&diffs);
    let (m, sd) = (s.mean, s.sd);
    let tost = if eq > 0.0 && n >= 2 && sd > 1e-15 {
        let se = sd / (n as f64).sqrt();
        one_sided_p((m + eq) / se, n - 1).max(one_sided_p((eq - m) / se, n - 1))
    } else if eq > 0.0 && sd <= 1e-15 {
        if m.abs() < eq { 0.0 } else { 1.0 }
    } else {
        f64::NAN
    };
    let p = if sd <= 1e-15 { if m.abs() < 1e-15 { 1.0 } else { 0.0 } } else { paired_t_pvalue(a, b) };
    Cmp { diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, p, tost_p: tost, n }
}
pub fn holm(ps: &[f64]) -> Vec<f64> {
    holm_adjust(ps)
}
pub fn mark(adj: f64, d: f64) -> &'static str {
    if adj < 0.05 { if d > 0.0 { "WIN" } else { "LOSS" } } else { "ns" }
}
pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}
pub fn jnum(x: f64) -> Value {
    if x.is_finite() { json!(x) } else { Value::Null }
}
