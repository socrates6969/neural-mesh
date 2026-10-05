//! g21 pipeline: the spec comes from the EXTRACTOR (not given clean), the verifier is the shared one or the independent
//! one, tools may carry a seeded evaluator bug, selection may use extractor confidence, abstain => plain path.
use std::collections::HashMap;

use neural_mesh::router::{Rng, Task};
use neural_mesh::tools::{tool_features, LookupTool, Registry, Sandbox, ToolSelector};

use super::evalx::{self, MCalc, MCsp, MIdentity, MUnits, MVerifier};
use super::extract::{extract, Ex};
use super::g13::*;
use super::indep::IVerifier;
use super::nlgen::{render_world, NlCfg, NlTask};

pub struct XWorld {
    pub w: World,
    pub nl: Vec<NlTask>,
    pub ex: Vec<Ex>,
}

pub fn build_xworld(cfg: &WorldCfg, ncfg: &NlCfg, seed: u64) -> XWorld {
    let w = gen_world(cfg, seed);
    let nl = render_world(&w.tasks, ncfg, seed);
    let ex: Vec<Ex> = nl.iter().map(|n| extract(&n.text)).collect();
    XWorld { w, nl, ex }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum XSel {
    /// keyword rule picks the tool (prose keywords -> none); the spec is whatever the extractor produced
    Static,
    /// the extractor's own kind guess if its confidence >= theta, else none
    Rule(f64),
    /// learned selector over shape tokens of the extracted spec (+ confidence bin + kind guess when `conf`)
    Learned { conf: bool },
    /// POST-HOC variant (designed after the held-out run of the pre-declared arms): learned(conf) + warm-start prior "extractor kind guess -> its tool"
    LearnedKg,
    /// the true kind's tool (none for qual) when the extractor produced a spec
    Oracle,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ver {
    None,
    Shared,
    Indep { meta: bool },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Src {
    Extract,
    /// the canonical spec is given (upper bound; no extractor errors, no abstention)
    Clean,
}
#[derive(Clone, Debug)]
pub struct XArm {
    pub name: String,
    pub router: Rt,
    pub sel: XSel,
    pub ver: Ver,
    pub src: Src,
    pub cache: bool,
    pub memo: bool,
}
impl XArm {
    pub fn new(name: &str, sel: XSel, ver: Ver, src: Src, cache: bool, memo: bool) -> Self {
        XArm { name: name.into(), router: Rt::Safe, sel, ver, src, cache, memo }
    }
}

#[derive(Clone, Debug, Default)]
pub struct XRun {
    pub st: RunStats,
    /// per compute task: the extractor/selector did not use a tool (abstain or no spec)
    pub no_tool: f64,
    /// per compute task: a tool answer was delivered AND it was wrong (silent error)
    pub silent_wrong: f64,
    /// per compute task: a tool answer was delivered and it was right
    pub tool_right: f64,
    /// verifier rejections per task
    pub rejects: f64,
    /// fallbacks (tool failed / rejected) per compute task
    pub fallback: f64,
    pub ex_cost: f64,
    pub ver_cost: f64,
}

enum Att {
    Accepted { ans: String, verified: bool },
    Failed,
    Rejected,
    Ignored,
}

struct Ctx<'a> {
    w: &'a World,
    reg: Registry,
    cost: f64,
    nominal: f64,
    tool_calls: u64,
    verify_calls: u64,
    verify_reject: u64,
    ver_cost: f64,
}

fn make_reg(lookup: &LookupTool, cache: bool, ver: Ver) -> Registry {
    let mut r = Registry::new(cache);
    r.register(Box::new(MCalc));
    r.register(Box::new(MUnits));
    r.register(Box::new(MCsp));
    r.register(Box::new(MIdentity::default()));
    r.register(Box::new(Sandbox::default()));
    r.register(Box::new(lookup.clone()));
    match ver {
        Ver::Indep { meta } => r.register(Box::new(IVerifier::new(lookup.clone(), meta))),
        _ => r.register(Box::new(MVerifier::new(lookup.clone()))),
    };
    r
}

fn direct(w: &World, t: &TaskRec, a: usize) -> String {
    if t.u_dir[a] < w.p_dir(a, t) { t.truth.correct_string() } else { wrong_answer(t, t.salt ^ ((a as u64 + 1) * 0x9E37)) }
}

fn conf_bin(c: f64) -> &'static str {
    if c < 0.3 {
        "cf:lo"
    } else if c < 0.6 {
        "cf:mid"
    } else if c < 0.8 {
        "cf:hi"
    } else {
        "cf:top"
    }
}

fn tool_path(cx: &mut Ctx, t: &TaskRec, spec: &str, tool: usize, verify: bool) -> Att {
    // the agent presents / uses the tool result: one model call (conservative: the extractor replaced the
    // agent's formulation, but the final reply still costs a call)
    cx.cost += MODEL_COST;
    cx.nominal += MODEL_COST;
    cx.tool_calls += 1;
    let o = cx.reg.call_idx(tool, spec);
    cx.cost += o.cost as f64;
    cx.nominal += o.nominal as f64;
    let out = match o.result {
        Ok(s) => s,
        Err(_) => return Att::Failed,
    };
    if t.kind == Kind::Qual {
        return if t.u_trust < cx.w.cfg.trust_h { Att::Accepted { ans: format!("tool:{out}"), verified: false } } else { Att::Ignored };
    }
    if verify {
        let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], spec, out);
        cx.verify_calls += 1;
        let v = cx.reg.call_idx(VERIFY_IDX, &vi);
        cx.cost += v.cost as f64;
        cx.nominal += v.nominal as f64;
        cx.ver_cost += v.cost as f64;
        match v.result {
            Ok(s) if s == "accept" => Att::Accepted { ans: out, verified: true },
            _ => {
                cx.verify_reject += 1;
                Att::Rejected
            }
        }
    } else {
        Att::Accepted { ans: out, verified: false }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XCfg {
    pub ex_cost: f64,
    pub theta_floor: f64,
}
impl Default for XCfg {
    fn default() -> Self {
        XCfg { ex_cost: 0.3, theta_floor: 0.0 }
    }
}

pub fn run_x(xw: &XWorld, arm: &XArm, h: &Hyper, xc: &XCfg, seed: u64, mutant: u8) -> XRun {
    evalx::set_mutant(mutant);
    let r = run_x_inner(xw, arm, h, xc, seed);
    evalx::set_mutant(0);
    r
}

fn run_x_inner(xw: &XWorld, arm: &XArm, h: &Hyper, xc: &XCfg, seed: u64) -> XRun {
    let w = &xw.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut router = mk_router(arm.router, w, h);
    let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(5));
    let mut cx = Ctx { w, reg: make_reg(&w.lookup, arm.cache, arm.ver), cost: 0.0, nominal: 0.0, tool_calls: 0, verify_calls: 0, verify_reject: 0, ver_cost: 0.0 };
    let mut selector = ToolSelector::new(7, h.sel_eps, h.sel_shrink, h.sel_opt, seed.wrapping_mul(17).wrapping_add(3));
    if h.kw_prior > 0.0 {
        for k in KINDS {
            for kw in k.kws() {
                selector.prior(kw, static_tool(kw, true), h.kw_prior, 0.8);
            }
        }
    }
    if arm.sel == XSel::LearnedKg {
        for k in KINDS {
            if let Some(t) = k.tool() {
                selector.prior(&format!("kg:{}", k.label()), t, 20.0, 0.8);
            }
        }
    }
    let mut memo: HashMap<String, String> = HashMap::new();
    let mut st = RunStats::default();
    let (mut c_all, mut c_comp, mut c_qual) = (0.0, 0.0, 0.0);
    let (mut memo_hits, mut skip_q, mut right_c, mut used_c) = (0u64, 0u64, 0u64, 0u64);
    let (mut n_no_tool, mut n_silent, mut n_tool_right, mut n_fallback, mut ex_cost_tot) = (0u64, 0u64, 0u64, 0u64, 0.0f64);
    let verify = arm.ver != Ver::None;
    for (i, t) in w.tasks.iter().enumerate() {
        let is_q = t.kind == Kind::Qual;
        if is_q {
            st.n_qual += 1;
        } else {
            st.n_compute += 1;
        }
        let task = Task { id: t.id, task_type: None, features: t.rfeat.clone() };
        let nominal_before = cx.nominal;
        // the spec the pipeline works with
        let ex = &xw.ex[i];
        let (spec, conf, kind_guess): (Option<String>, f64, Option<Kind>) = match arm.src {
            Src::Extract => {
                cx.cost += xc.ex_cost;
                cx.nominal += xc.ex_cost;
                ex_cost_tot += xc.ex_cost;
                (ex.spec.clone(), ex.conf, ex.kind)
            }
            Src::Clean => (if is_q { None } else { Some(t.spec.clone()) }, 1.0, Some(t.kind)),
        };
        // memo
        if arm.memo {
            if let Some(key) = &spec {
                if let Some(ans) = memo.get(key) {
                    let ok = t.truth.is_correct(ans);
                    cx.cost += MEMO_COST;
                    cx.nominal += MEMO_COST;
                    memo_hits += 1;
                    if ok {
                        c_all += 1.0;
                        if is_q { c_qual += 1.0 } else { c_comp += 1.0 }
                    }
                    if is_q {
                        skip_q += 1;
                    } else {
                        used_c += 1;
                        right_c += 1;
                        if ok { n_tool_right += 1 } else { n_silent += 1 }
                    }
                    st.correct_vec.push(ok as u8);
                    continue;
                }
            }
        }
        let a1 = router.select(&task, &cands, &mut rng);
        // ---- tool choice
        let sfeat: Vec<String> = {
            let mut f = match &spec {
                Some(s) => tool_features(s),
                None => vec!["sh:none".to_string()],
            };
            if matches!(arm.sel, XSel::Learned { conf: true } | XSel::LearnedKg) {
                f.push(conf_bin(conf).to_string());
                if let Some(k) = kind_guess {
                    f.push(format!("kg:{}", k.label()));
                }
            }
            f.push(t.kw.clone());
            f
        };
        let choice = match (&spec, arm.sel) {
            (None, _) => NONE_ARM,
            (Some(_), XSel::Static) => static_tool(&t.kw, true),
            (Some(_), XSel::Rule(theta)) => {
                if conf >= theta.max(xc.theta_floor) {
                    kind_guess.and_then(|k| k.tool()).unwrap_or(NONE_ARM)
                } else {
                    NONE_ARM
                }
            }
            (Some(_), XSel::Oracle) => t.kind.tool().unwrap_or(NONE_ARM),
            (Some(_), XSel::Learned { .. } | XSel::LearnedKg) => selector.select(&sfeat),
        };
        let ok: bool;
        let mut sel_score = 0.0f64;
        if choice == NONE_ARM {
            let ans = direct(w, t, a1);
            cx.cost += MODEL_COST;
            cx.nominal += MODEL_COST;
            ok = t.truth.is_correct(&ans);
            sel_score = ok as u8 as f64;
            router.update(&task, a1, ok as u8 as f64);
            if is_q {
                skip_q += 1;
            } else {
                n_no_tool += 1;
            }
        } else {
            let spec_s = spec.as_deref().unwrap_or("");
            if !is_q {
                used_c += 1;
                if t.kind.tool() == Some(choice) {
                    right_c += 1;
                }
            }
            let fin: String;
            let mut ver_ok = false;
            let mut delivered_tool = false;
            let mut fit = 0.0f64;
            let mut credit_a1 = false;
            match tool_path(&mut cx, t, spec_s, choice, verify) {
                Att::Accepted { ans, verified } => {
                    ver_ok = verified;
                    delivered_tool = true;
                    credit_a1 = true;
                    fit = if verified { 1.0 } else { -1.0 }; // -1: use ground truth below
                    fin = ans;
                }
                Att::Ignored => {
                    fin = direct(w, t, a1);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    credit_a1 = true;
                    fit = -1.0;
                }
                Att::Failed | Att::Rejected => {
                    // deterministic extractor: re-asking cannot change the spec; fall back to another agent's direct answer
                    if !is_q {
                        n_fallback += 1;
                    }
                    let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                    let a2 = router.select(&task, &c2, &mut rng);
                    fin = direct(w, t, a2);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    let okf = t.truth.is_correct(&fin);
                    router.update(&task, a2, okf as u8 as f64);
                }
            }
            ok = t.truth.is_correct(&fin);
            if credit_a1 {
                // credit the first agent with the final outcome when its path produced the answer
                router.update(&task, a1, ok as u8 as f64);
            }
            if delivered_tool && !is_q {
                if ok { n_tool_right += 1 } else { n_silent += 1 }
            }
            sel_score = if fit > 0.0 { fit } else if fit < 0.0 { ok as u8 as f64 } else { 0.0 };
            sel_score = 0.5 * sel_score + 0.5 * (ok as u8 as f64);
            if arm.memo && ver_ok && ok {
                if let Some(key) = &spec {
                    memo.insert(key.clone(), fin.clone());
                }
            }
        }
        if matches!(arm.sel, XSel::Learned { .. } | XSel::LearnedKg) {
            if spec.is_some() {
                let spent = (cx.nominal - nominal_before) / MODEL_COST;
                selector.update(&sfeat, choice, sel_score - h.kappa * spent);
            }
        }
        if ok {
            c_all += 1.0;
            if is_q { c_qual += 1.0 } else { c_comp += 1.0 }
        }
        st.correct_vec.push(ok as u8);
    }
    let n = w.tasks.len() as f64;
    let nc = st.n_compute.max(1) as f64;
    st.correct = c_all / n;
    st.correct_compute = if st.n_compute > 0 { c_comp / st.n_compute as f64 } else { f64::NAN };
    st.correct_qual = if st.n_qual > 0 { c_qual / st.n_qual as f64 } else { f64::NAN };
    st.cost = cx.cost / n;
    st.cost_nominal = cx.nominal / n;
    st.tool_calls = cx.tool_calls as f64 / n;
    st.verify_calls = cx.verify_calls as f64 / n;
    st.verify_reject = if cx.verify_calls > 0 { cx.verify_reject as f64 / cx.verify_calls as f64 } else { 0.0 };
    st.cache_hit = cx.reg.stats.hit_rate();
    st.cache_saved = cx.reg.stats.cost_saved as f64 / n;
    st.memo_hit = memo_hits as f64 / n;
    st.skip_qual = if st.n_qual > 0 { skip_q as f64 / st.n_qual as f64 } else { f64::NAN };
    st.right_tool = if st.n_compute > 0 { right_c as f64 / nc } else { f64::NAN };
    st.tool_used_compute = if st.n_compute > 0 { used_c as f64 / nc } else { f64::NAN };
    XRun {
        st,
        no_tool: n_no_tool as f64 / nc,
        silent_wrong: n_silent as f64 / nc,
        tool_right: n_tool_right as f64 / nc,
        rejects: cx.verify_reject as f64 / n,
        fallback: n_fallback as f64 / nc,
        ex_cost: ex_cost_tot / n,
        ver_cost: cx.ver_cost / n,
    }
}
