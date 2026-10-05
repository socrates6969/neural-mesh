//! h18 seeded worlds (two independent generators A / B), the "ungated model" assumption, routers and pipeline arms.
//!
//! ASSUMPTIONS (stated so nobody reads more into the numbers than is there; all synthetic until validated on real traffic):
//!  * the UNGATED model answers a math task correctly with probability (1-e)^steps, e = err[agent][topic] * err_scale *
//!    (1 + sens[agent] * curve(difficulty)): per-step error rises with difficulty, and the success probability falls
//!    geometrically with the number of by-hand steps. `err_scale` is swept in `sens`;
//!  * a wrong direct answer is the true value perturbed (digit slip / scale / sign), never accidentally right;
//!  * when a quantity was deleted from the text (level 4 "dropped") the model can only guess: success x0.25;
//!  * tool offload formulation is done by the rule-based extractor (real code, real errors); the final reply is still
//!    written by the routed agent (cost 10 per answered task);
//!  * verifiers are the real fin.rs verifiers, run on the EXTRACTED spec: they can catch evaluator bugs and wrong model
//!    claims but not extraction errors (same spec).
use super::ext::{conf_bin, extract, Ex};
use super::nl::*;
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::fin::{self, DomainVerifier, EconTool, FinTool, SciTool};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task as RTask};
use neural_mesh::tools::{Tool, ToolSelector};
use std::collections::{HashMap, HashSet};

pub const NA: usize = 6;
pub const NSUB: usize = 6;
pub const MODEL_COST: f64 = 10.0;
pub const EXTRACT_COST: f64 = 0.3;
pub const NONE_ARM: usize = 3;

#[derive(Clone, Debug)]
pub struct WCfg {
    pub name: &'static str,
    pub style: u8,
    pub n: usize,
    /// finance, business/econ, science, no-math prose
    pub dom_p: [f64; 4],
    pub lvl_p: [f64; 5],
    pub unseen_p: f64,
    pub misuse: f64,
    pub dup: f64,
    pub err_scale: f64,
}
impl WCfg {
    pub fn world_a() -> Self {
        WCfg { name: "A", style: 0, n: 600, dom_p: [0.33, 0.30, 0.27, 0.10], lvl_p: [0.30, 0.30, 0.22, 0.13, 0.05], unseen_p: 0.20, misuse: 0.30, dup: 0.15, err_scale: 1.0 }
    }
    pub fn world_b() -> Self {
        WCfg { name: "B", style: 1, n: 600, dom_p: [0.20, 0.25, 0.38, 0.17], lvl_p: [0.12, 0.22, 0.26, 0.30, 0.10], unseen_p: 0.35, misuse: 0.25, dup: 0.20, err_scale: 1.0 }
    }
    pub fn prose_only(mut self) -> Self {
        self.dom_p = [0.0, 0.0, 0.0, 1.0];
        self
    }
}

#[derive(Clone, Debug)]
pub struct Task {
    pub id: usize,
    /// 0 finance, 1 business, 2 science, 3 prose
    pub dom: usize,
    pub sub: usize,
    pub op: &'static str,
    pub spec: String,
    pub truth: String,
    pub d: f64,
    pub steps: u32,
    pub text: String,
    pub level: u8,
    pub unseen: bool,
    pub dropped: bool,
    pub kw: String,
    pub rfeat: Vec<String>,
    pub u_dir: [f64; NA],
    pub salt: u64,
    pub dup: bool,
    pub ex: Ex,
}

#[derive(Clone, Debug)]
pub struct Agents {
    pub err: [[f64; NSUB]; NA],
    pub sens: [f64; NA],
    pub qual: [f64; NA],
}
pub struct World {
    pub cfg: WCfg,
    pub tasks: Vec<Task>,
    pub agents: Agents,
    pub kwmap: HashMap<String, usize>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
impl World {
    pub fn curve(&self, d: f64) -> f64 {
        if self.cfg.style == 0 {
            d.powf(1.3)
        } else {
            (sigmoid(7.0 * (d - 0.5)) - sigmoid(-3.5)) / (sigmoid(3.5) - sigmoid(-3.5))
        }
    }
    pub fn p_dir(&self, a: usize, t: &Task) -> f64 {
        if t.dom == 3 {
            return self.agents.qual[a];
        }
        let e = (self.agents.err[a][t.sub] * self.cfg.err_scale * (1.0 + self.agents.sens[a] * self.curve(t.d))).clamp(0.0005, 0.6);
        let mut p = (1.0 - e).powi(t.steps as i32);
        if t.dropped {
            p *= 0.25;
        }
        p.clamp(0.02, 0.99)
    }
}

pub fn kw_of(op: &str) -> &'static str {
    OPS.iter().find(|o| o.op == op).map(|o| o.op).unwrap_or("explain")
}
pub const PROSE_KW: [&str; 4] = ["explain", "discuss", "summarise", "which"];
/// static keyword -> tool family (0 fin, 1 sci, 2 econ) or NONE_ARM
pub fn static_tool(kw: &str) -> usize {
    match fin::tool_of(kw) {
        Some(t) => t,
        None => NONE_ARM,
    }
}

fn wrong_answer(t: &Task, salt: u64) -> String {
    let mut r = Rng::new(salt ^ 0xA5A5_1234);
    if t.dom == 3 {
        let letters = ['A', 'B', 'C', 'D'];
        let tl = t.truth.chars().next().unwrap_or('A');
        let c = letters.iter().cloned().filter(|c| *c != tl).collect::<Vec<_>>();
        return c[r.below(c.len())].to_string();
    }
    match t.truth.parse::<f64>() {
        Ok(x) if t.op != "sfround" && t.op != "sci" => {
            for _ in 0..8 {
                let w = match r.below(5) {
                    0 => x * (1.0 + 0.02 + 0.3 * r.f64()),
                    1 => x * (1.0 - 0.02 - 0.3 * r.f64()),
                    2 => x * 10f64.powi(if r.f64() < 0.5 { 1 } else { -1 }),
                    3 => -x + 0.01 * (1.0 + r.f64()),
                    _ => x + (1.0 + 9.0 * r.f64()) * 10f64.powf((x.abs().max(1e-6)).log10().floor() - 1.0),
                };
                let s = fin::fmt((w * 1e7).round() / 1e7);
                if let Ok(wf) = s.parse::<f64>() {
                    if !fin::close(wf, x) {
                        return s;
                    }
                }
            }
            fin::fmt(x * 1.5 + 1.0)
        }
        _ => {
            // text answer: change a digit
            let mut cs: Vec<char> = t.truth.chars().collect();
            let idx: Vec<usize> = cs.iter().enumerate().filter(|(_, c)| c.is_ascii_digit()).map(|(i, _)| i).collect();
            if idx.is_empty() {
                return format!("{}x", t.truth);
            }
            let i = idx[r.below(idx.len())];
            let d = cs[i].to_digit(10).unwrap_or(0);
            cs[i] = char::from_digit((d + 1 + r.below(8) as u32) % 10, 10).unwrap_or('0');
            cs.into_iter().collect()
        }
    }
}

pub fn is_correct(t: &Task, ans: &str) -> bool {
    if t.dom == 3 || t.op == "sfround" || t.op == "sci" {
        return ans == t.truth;
    }
    match (t.truth.parse::<f64>(), ans.parse::<f64>()) {
        (Ok(a), Ok(b)) => fin::close(a, b),
        _ => ans == t.truth,
    }
}

/// answer of the UNGATED model `a` (deterministic per (task, agent))
pub fn direct(w: &World, t: &Task, a: usize) -> String {
    if t.u_dir[a] < w.p_dir(a, t) {
        t.truth.clone()
    } else {
        wrong_answer(t, t.salt.wrapping_add(a as u64 * 7919))
    }
}

fn tool_for(idx: usize, bug: u8) -> Box<dyn Tool> {
    match idx {
        0 => Box::new(FinTool { bug }),
        1 => Box::new(SciTool { bug }),
        _ => Box::new(EconTool { bug }),
    }
}
pub fn op_of_spec(spec: &str) -> &str {
    spec.split_whitespace().next().unwrap_or("")
}

pub fn gen_world(cfg: &WCfg, seed: u64) -> World {
    let mut r = Rng::new(seed.wrapping_mul(104_729).wrapping_add(cfg.style as u64 * 7_919 + 31));
    let mut ag = Agents { err: [[0.0; NSUB]; NA], sens: [1.0; NA], qual: [0.8; NA] };
    for a in 0..NA {
        if cfg.style == 0 {
            for g in 0..NSUB {
                ag.err[a][g] = 0.010 + 0.035 * r.f64();
            }
            ag.sens[a] = 1.0 + 1.2 * r.f64();
            ag.qual[a] = 0.75 + 0.2 * r.f64();
        } else {
            let s = 0.4 + 0.6 * r.f64();
            let special = r.below(NSUB);
            for g in 0..NSUB {
                ag.err[a][g] = (0.075 - 0.055 * s - if g == special { 0.012 } else { 0.0 } + 0.004 * r.f64()).max(0.004);
            }
            ag.sens[a] = 0.5 + 0.9 * r.f64();
            ag.qual[a] = 0.70 + 0.25 * s;
        }
    }
    let mut kwmap = HashMap::new();
    for g in 0..NSUB {
        // believed best agent for the topic (70% right): lowest mean error
        let best = (0..NA).fold(0, |b, a| if ag.err[a][g] < ag.err[b][g] { a } else { b });
        let pick = if r.f64() < 0.7 { best } else { r.below(NA) };
        kwmap.insert(format!("dom{g}"), pick);
    }
    let mut tasks: Vec<Task> = Vec::with_capacity(cfg.n);
    let ops_by_dom: Vec<Vec<&'static OpDef>> = (0..3).map(|d| OPS.iter().filter(|o| o.dom == d).collect()).collect();
    let dsum: f64 = cfg.dom_p.iter().sum();
    let lsum: f64 = cfg.lvl_p.iter().sum();
    let mut i = 0;
    while i < cfg.n {
        let mut draws = [0.0; NA];
        for a in 0..NA {
            draws[a] = r.f64();
        }
        let salt = r.next_u64();
        if i > 8 && r.f64() < cfg.dup {
            let back = 1 + (r.f64().powi(2) * (i.min(120) as f64 - 1.0)) as usize;
            let mut t = tasks[i - back].clone();
            t.id = i;
            t.u_dir = draws;
            t.salt = salt;
            t.dup = true;
            tasks.push(t);
            i += 1;
            continue;
        }
        let mut u = r.f64() * dsum;
        let mut dom = 3;
        for (k, p) in cfg.dom_p.iter().enumerate() {
            if u < *p {
                dom = k;
                break;
            }
            u -= *p;
        }
        let unseen = r.f64() < cfg.unseen_p;
        let mut ul = r.f64() * lsum;
        let mut level = 0u8;
        for (k, p) in cfg.lvl_p.iter().enumerate() {
            if ul < *p {
                level = k as u8;
                break;
            }
            ul -= *p;
        }
        let d = if cfg.style == 0 { r.f64() } else { r.f64().powf(0.7) };
        if dom == 3 {
            let text = gen_prose(&mut r, unseen);
            let kw = PROSE_KW[r.below(PROSE_KW.len())].to_string();
            let sub = r.below(NSUB);
            let ex = extract(&text);
            let rfeat = vec![format!("dom{sub}"), kw.clone(), format!("nz{}", r.below(8))];
            let truth = ['A', 'B', 'C', 'D'][r.below(4)].to_string();
            tasks.push(Task { id: i, dom, sub, op: "", spec: String::new(), truth, d, steps: 1, text, level: 0, unseen, dropped: false, kw, rfeat, u_dir: draws, salt, dup: false, ex });
            i += 1;
            continue;
        }
        let pool = &ops_by_dom[dom];
        let def = pool[r.below(pool.len())];
        let params = match gen_params(def.op, &mut r, d, cfg.style) {
            Some(p) => p,
            None => continue,
        };
        let spec = params.spec();
        let truth = match FinTool::default().call(&spec).or_else(|_| SciTool::default().call(&spec)).or_else(|_| EconTool::default().call(&spec)) {
            Ok(s) => fin::parse_out(&s).get("v").cloned().unwrap_or_default(),
            Err(_) => continue,
        };
        let (text, dropped) = render(def, &params, &RCfg { level, unseen }, &mut r);
        let kw_op = if r.f64() < cfg.misuse {
            let mut o = OPS[r.below(OPS.len())].op;
            while fin::tool_of(o) == fin::tool_of(def.op) {
                o = OPS[r.below(OPS.len())].op;
            }
            o
        } else {
            def.op
        };
        let kw = kw_of(kw_op).to_string();
        let rfeat = vec![format!("dom{}", def.sub), kw.clone(), format!("nz{}", r.below(8))];
        let ex = extract(&text);
        tasks.push(Task { id: i, dom, sub: def.sub, op: def.op, spec, truth, d, steps: params.steps, text, level, unseen, dropped, kw, rfeat, u_dir: draws, salt, dup: false, ex });
        i += 1;
    }
    World { cfg: cfg.clone(), tasks, agents: ag, kwmap }
}

// ---------------------------------------------------------------------------------------------
// routers

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rt {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Safe,
}
pub const ROUTERS: [(Rt, &str); 6] = [(Rt::Keyword, "keyword"), (Rt::Eps, "eps-greedy"), (Rt::Pts, "pooled-thompson"), (Rt::HebR, "hebbian-robust"), (Rt::Warm, "warm(hebr)"), (Rt::Safe, "guarded-safe")];

#[derive(Clone, Copy, Debug)]
pub struct Hyper {
    pub eps_g: f64,
    pub eps_e: f64,
    pub pts_g: f64,
    pub warm_k: f64,
    pub sel_eps: f64,
    pub sel_shrink: f64,
    pub sel_opt: f64,
    pub kappa: f64,
    pub kw_prior: f64,
    pub theta: f64,
    pub theta_lo: f64,
    /// weight of the verifier-fit term in the selector reward (rest = graded correctness)
    pub fit_w: f64,
}
impl Default for Hyper {
    fn default() -> Self {
        Hyper { eps_g: 1.0, eps_e: 0.1, pts_g: 1.0, warm_k: 10.0, sel_eps: 0.02, sel_shrink: 4.0, sel_opt: 0.7, kappa: 0.04, kw_prior: 20.0, theta: 0.4, theta_lo: 0.1, fit_w: 0.5 }
    }
}
pub fn mk_router(rt: Rt, w: &World, h: &Hyper) -> Box<dyn Router> {
    match rt {
        Rt::Keyword => Box::new(KeywordRouter::new(w.kwmap.clone(), NA)),
        Rt::Eps => Box::new(EpsGreedyRouter::new(NA, h.eps_g, h.eps_e)),
        Rt::Pts => Box::new(PooledThompsonRouter::new(NA, h.pts_g)),
        Rt::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Rt::Warm => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), w.kwmap.clone(), NA, h.warm_k).with_evidence(Evidence::MeanFeature)),
        Rt::Safe => Box::new(recommended_router(w.kwmap.clone(), NA)),
    }
}

// ---------------------------------------------------------------------------------------------
// arms

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sel {
    /// use the tool whenever a spec exists
    Always,
    /// use the tool when extractor confidence >= theta
    Rule,
    Learned,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ver {
    None,
    Shared,
    Indep,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    Extracted,
    Canonical,
}
#[derive(Clone, Debug)]
pub enum Mode {
    Plain,
    /// the model answers, a verifier checks the claim against the extracted spec, reject -> another agent
    VOnly { indep: bool },
    Vote3,
    Tool { src: Src, sel: Sel, ver: Ver, cross: bool, cache: bool },
    /// POST-HOC (designed on tune seeds 1-8 after the pre-declared headline H): confidence tiers on the extractor.
    /// conf >= hi -> tool + independent verifier; lo <= conf < hi -> V-only (verify the model's direct answer against the
    /// extracted spec, reject -> another agent); else plain.
    Tier,
}
#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    pub router: Rt,
    pub mode: Mode,
}
impl Arm {
    pub fn plain(rt: Rt, name: &str) -> Arm {
        Arm { name: name.to_string(), router: rt, mode: Mode::Plain }
    }
    pub fn tool(name: &str, src: Src, sel: Sel, ver: Ver, cross: bool, cache: bool) -> Arm {
        Arm { name: name.to_string(), router: Rt::Safe, mode: Mode::Tool { src, sel, ver, cross, cache } }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub correct: f64,
    pub correct_dom: [f64; 4],
    pub n_dom: [usize; 4],
    pub cost: f64,
    pub cost_nominal: f64,
    pub tool_rate_math: f64,
    pub tool_rate_prose: f64,
    pub silent_wrong: f64,
    pub ver_reject: f64,
    pub cache_hit: f64,
    pub right_family: f64,
    pub correct_vec: Vec<u8>,
}

struct Cx {
    cost: f64,
    nominal: f64,
    cache: bool,
    seen: HashMap<(u8, String), (Result<String, String>, u32)>,
    calls: u64,
    hits: u64,
    ver_calls: u64,
    ver_rej: u64,
}
impl Cx {
    fn tool(&mut self, idx: usize, bug: u8, spec: &str) -> Result<String, String> {
        self.calls += 1;
        let key = (idx as u8, spec.to_string());
        if self.cache {
            if let Some((r, c)) = self.seen.get(&key) {
                self.hits += 1;
                self.nominal += *c as f64;
                return r.clone();
            }
        }
        let t = tool_for(idx, bug);
        let (res, cost) = t.call_metered(spec);
        let r = res.map(|s| fin::parse_out(&s).get("v").cloned().unwrap_or_default()).map_err(|e| e.to_string());
        self.cost += cost as f64;
        self.nominal += cost as f64;
        if self.cache {
            self.seen.insert(key, (r.clone(), cost));
        }
        r
    }
    /// accept?
    fn verify(&mut self, indep: bool, bug: u8, spec: &str, claim: &str) -> bool {
        self.ver_calls += 1;
        let key = (if indep { 10 } else { 11 }, format!("{spec} => {claim}"));
        let (acc, cost) = if self.cache && self.seen.contains_key(&key) {
            self.hits += 1;
            let c = self.seen[&key].1;
            self.nominal += c as f64;
            (self.seen[&key].0 == Ok("accept".to_string()), 0u32)
        } else {
            let v = DomainVerifier { indep, bug };
            let (r, c) = (v.call(&format!("{spec} => {claim}")), v.cost());
            let acc = matches!(&r, Ok(s) if s == "accept");
            self.nominal += c as f64;
            if self.cache {
                self.seen.insert(key, (if acc { Ok("accept".to_string()) } else { Err("reject".to_string()) }, c));
            }
            (acc, c)
        };
        self.cost += cost as f64;
        if !acc {
            self.ver_rej += 1;
        }
        acc
    }
}

fn agree(a: &str, b: &str) -> bool {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) => fin::close(x, y),
        _ => a == b,
    }
}

pub fn run_arm(w: &World, arm: &Arm, h: &Hyper, seed: u64, bug: u8) -> RunStats {
    let cands: Vec<usize> = (0..NA).collect();
    let mut router = mk_router(arm.router, w, h);
    let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(5));
    let cache = matches!(&arm.mode, Mode::Tool { cache: true, .. }) || arm.name.ends_with("+cache");
    let mut cx = Cx { cost: 0.0, nominal: 0.0, cache, seen: HashMap::new(), calls: 0, hits: 0, ver_calls: 0, ver_rej: 0 };
    let mut selector = ToolSelector::new(4, h.sel_eps, h.sel_shrink, h.sel_opt, seed.wrapping_mul(17).wrapping_add(3));
    if h.kw_prior > 0.0 {
        for o in OPS {
            selector.prior(o.op, static_tool(o.op), h.kw_prior, 0.8);
        }
        for k in PROSE_KW {
            selector.prior(k, NONE_ARM, h.kw_prior, 0.8);
        }
    }
    let mut st = RunStats::default();
    let (mut tool_math, mut n_math, mut tool_prose, mut n_prose, mut silent, mut fam_right, mut fam_n) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut csum = [0.0f64; 4];
    for t in &w.tasks {
        let is_prose = t.dom == 3;
        if is_prose {
            n_prose += 1;
        } else {
            n_math += 1;
        }
        st.n_dom[t.dom] += 1;
        let task = rtask(t);
        let a1 = router.select(&task, &cands, &mut rng);
        let ok: bool;
        let mut pre_charged = false;
        let eff_mode: Mode = match &arm.mode {
            Mode::Tier => {
                cx.cost += EXTRACT_COST;
                cx.nominal += EXTRACT_COST;
                pre_charged = true;
                match &t.ex.spec {
                    Some(_) if t.ex.conf >= h.theta => Mode::Tool { src: Src::Extracted, sel: Sel::Always, ver: Ver::Indep, cross: false, cache: false },
                    Some(_) if t.ex.conf >= h.theta_lo => Mode::VOnly { indep: true },
                    _ => Mode::Plain,
                }
            }
            m => m.clone(),
        };
        match &eff_mode {
            Mode::Tier => unreachable!(),
            Mode::Plain => {
                let ans = direct(w, t, a1);
                cx.cost += MODEL_COST;
                cx.nominal += MODEL_COST;
                ok = is_correct(t, &ans);
                router.update(&task, a1, ok as u8 as f64);
            }
            Mode::Vote3 => {
                let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                let a2 = router.select(&task, &c2, &mut rng);
                let c3: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1 && *x != a2).collect();
                let a3 = router.select(&task, &c3, &mut rng);
                let ans: Vec<String> = [a1, a2, a3].iter().map(|a| direct(w, t, *a)).collect();
                cx.cost += 3.0 * MODEL_COST;
                cx.nominal += 3.0 * MODEL_COST;
                let fin_ans = if agree(&ans[1], &ans[2]) { ans[1].clone() } else { ans[0].clone() };
                ok = is_correct(t, &fin_ans);
                for (i, a) in [a1, a2, a3].iter().enumerate() {
                    router.update(&task, *a, is_correct(t, &ans[i]) as u8 as f64);
                }
            }
            Mode::VOnly { indep } => {
                let ans = direct(w, t, a1);
                cx.cost += MODEL_COST;
                cx.nominal += MODEL_COST;
                let mut fin_ans = ans.clone();
                let mut rejected = false;
                if let Some(spec) = &t.ex.spec {
                    if !pre_charged {
                        cx.cost += EXTRACT_COST;
                        cx.nominal += EXTRACT_COST;
                    }
                    if !cx.verify(*indep, 0, spec, &ans) {
                        rejected = true;
                    }
                }
                if rejected {
                    router.update(&task, a1, 0.0);
                    let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                    let a2 = router.select(&task, &c2, &mut rng);
                    fin_ans = direct(w, t, a2);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    ok = is_correct(t, &fin_ans);
                    router.update(&task, a2, ok as u8 as f64);
                } else {
                    ok = is_correct(t, &fin_ans);
                    router.update(&task, a1, ok as u8 as f64);
                }
            }
            Mode::Tool { src, sel, ver, cross, .. } => {
                let nominal_before = cx.nominal;
                let (spec_opt, conf, op_guess): (Option<String>, f64, String) = match src {
                    Src::Extracted => {
                        if !pre_charged {
                            cx.cost += EXTRACT_COST;
                            cx.nominal += EXTRACT_COST;
                        }
                        (t.ex.spec.clone(), t.ex.conf, t.ex.op.unwrap_or("").to_string())
                    }
                    Src::Canonical => {
                        if is_prose {
                            (None, 0.0, String::new())
                        } else {
                            (Some(t.spec.clone()), 1.0, t.op.to_string())
                        }
                    }
                };
                let sfeat: Vec<String> = {
                    let mut f = vec![t.kw.clone()];
                    match &spec_opt {
                        Some(_) => {
                            f.push(format!("op:{op_guess}"));
                            f.push(format!("cb{}", conf_bin(conf)));
                        }
                        None => f.push("nospec".to_string()),
                    }
                    f
                };
                let mut choice = NONE_ARM;
                let use_spec = if let Some(spec) = &spec_opt {
                    let fam = fin::tool_of(op_of_spec(spec));
                    match sel {
                        Sel::Always => {
                            choice = fam.unwrap_or(NONE_ARM);
                            true
                        }
                        Sel::Rule => {
                            choice = fam.unwrap_or(NONE_ARM);
                            conf >= h.theta
                        }
                        Sel::Learned => {
                            choice = selector.select(&sfeat);
                            choice != NONE_ARM
                        }
                    }
                } else {
                    false
                };
                let mut sel_score = 0.0;
                let mut fallback_credit: Vec<(usize, f64)> = Vec::new();
                let fin_ans: String;
                if !use_spec {
                    let ans = direct(w, t, a1);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    ok = is_correct(t, &ans);
                    sel_score = ok as u8 as f64;
                    fallback_credit.push((a1, ok as u8 as f64));
                    fin_ans = ans;
                } else {
                    if is_prose {
                        tool_prose += 1;
                    } else {
                        tool_math += 1;
                        fam_n += 1;
                        if let Some(sp) = &spec_opt {
                            if fin::tool_of(op_of_spec(sp)) == Some(choice) && op_of_spec(sp) == t.op {
                                fam_right += 1;
                            }
                        }
                    }
                    let spec = spec_opt.clone().unwrap_or_default();
                    // the routed agent writes the final reply
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    let res = cx.tool(choice.min(2), bug, &spec);
                    let mut answer: Option<String> = None;
                    let mut fit = 0.0;
                    match res {
                        Ok(out) => {
                            let accepted = match ver {
                                Ver::None => true,
                                Ver::Shared => cx.verify(false, bug, &spec, &out),
                                Ver::Indep => cx.verify(true, if neural_mesh::fin::bug::is_common_mode(bug) { bug } else { 0 }, &spec, &out),
                            };
                            if accepted {
                                answer = Some(out);
                                fit = 1.0;
                            }
                        }
                        Err(_) => {}
                    }
                    let mut credited_a1 = true;
                    let mut ans_final: String;
                    match answer {
                        Some(a) => {
                            ans_final = a;
                            if *cross {
                                // cross-check against the model's own direct answer
                                let d1 = direct(w, t, a1);
                                cx.cost += MODEL_COST;
                                cx.nominal += MODEL_COST;
                                if !agree(&d1, &ans_final) {
                                    let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                                    let a2 = router.select(&task, &c2, &mut rng);
                                    let d2 = direct(w, t, a2);
                                    cx.cost += MODEL_COST;
                                    cx.nominal += MODEL_COST;
                                    if agree(&d2, &ans_final) {
                                    } else if agree(&d2, &d1) {
                                        ans_final = d1;
                                    } else if conf < 0.6 {
                                        ans_final = d1;
                                    }
                                }
                            }
                        }
                        None => {
                            // tool error or verifier rejection: another agent answers directly
                            fallback_credit.push((a1, 0.0));
                            credited_a1 = false;
                            let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                            let a2 = router.select(&task, &c2, &mut rng);
                            ans_final = direct(w, t, a2);
                            cx.cost += MODEL_COST;
                            cx.nominal += MODEL_COST;
                            let o2 = is_correct(t, &ans_final);
                            fallback_credit.push((a2, o2 as u8 as f64));
                        }
                    }
                    ok = is_correct(t, &ans_final);
                    if credited_a1 {
                        fallback_credit.push((a1, ok as u8 as f64));
                    }
                    if fit == 1.0 && !ok && !is_prose {
                        silent += 1;
                    }
                    sel_score = h.fit_w * fit + (1.0 - h.fit_w) * (ok as u8 as f64);
                    fin_ans = ans_final;
                }
                let _ = &fin_ans;
                for (ag, rw) in &fallback_credit {
                    router.update(&task, *ag, *rw);
                }
                if *sel == Sel::Learned && spec_opt.is_some() {
                    let spent = (cx.nominal - nominal_before) / MODEL_COST;
                    selector.update(&sfeat, choice, sel_score - h.kappa * spent);
                }
            }
        }
        if ok {
            csum[t.dom] += 1.0;
        }
        st.correct_vec.push(ok as u8);
    }
    let n = w.tasks.len() as f64;
    st.correct = csum.iter().sum::<f64>() / n;
    for g in 0..4 {
        st.correct_dom[g] = if st.n_dom[g] > 0 { csum[g] / st.n_dom[g] as f64 } else { f64::NAN };
    }
    st.cost = cx.cost / n;
    st.cost_nominal = cx.nominal / n;
    st.tool_rate_math = if n_math > 0 { tool_math as f64 / n_math as f64 } else { f64::NAN };
    st.tool_rate_prose = if n_prose > 0 { tool_prose as f64 / n_prose as f64 } else { f64::NAN };
    st.silent_wrong = if n_math > 0 { silent as f64 / n_math as f64 } else { f64::NAN };
    st.ver_reject = if cx.ver_calls > 0 { cx.ver_rej as f64 / cx.ver_calls as f64 } else { 0.0 };
    st.cache_hit = if cx.calls + cx.ver_calls > 0 { cx.hits as f64 / (cx.calls + cx.ver_calls) as f64 } else { 0.0 };
    st.right_family = if fam_n > 0 { fam_right as f64 / fam_n as f64 } else { f64::NAN };
    st
}

fn rtask(t: &Task) -> RTask {
    RTask { id: t.id as u64, task_type: None, features: t.rfeat.clone() }
}

/// value (`v`) of a spec line through the real tool at evaluator bug `bug`
pub fn eval_spec(spec: &str, bug: u8) -> Option<String> {
    let a = fin::Args::parse(spec).ok()?;
    let o = fin::eval(bug, &a).ok()?;
    Some(o.v)
}

pub fn set_of<T: std::hash::Hash + Eq + Clone>(v: &[T]) -> HashSet<T> {
    v.iter().cloned().collect()
}
