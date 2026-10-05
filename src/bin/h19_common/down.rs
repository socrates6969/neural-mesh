//! Downstream QUESTION-ANSWERING / metric-computation session simulation.
//! Plain reader (model reads the text directly) is a SEEDED NOISE MODEL (assumptions below, swept in sensitivity);
//! extraction arms compute answers EXACTLY from the extractor's values (so extractor errors become wrong answers),
//! verification arms re-read only flagged fields, retrieval arms restrict the text to BM25-selected blocks.
//! Everything is SYNTHETIC. The reader-noise constants are assumptions, not measurements of any real model.
use super::cal::*;
use super::gens::count_numtok;
use super::genb::METS;
use super::scan::*;
use super::verify::*;
use super::*;
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::hybrid::Bm25;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Router, Task};
use std::collections::HashMap;

pub const NAG: usize = 4;
pub const REVEAL: f64 = 0.6;
// ---- reader noise model (ASSUMPTIONS; all multiplied by SimCfg.noise)
pub const E0: f64 = 0.010; // base per-fact misread (x agent multiplier 0.5..1.6)
pub const A_DIS: f64 = 0.0004; // per distractor number in view (capped at 150)
pub const B_LEN: f64 = 0.006; // per ln(1 + chars/800)
pub const C_SCALE: f64 = 0.030; // scale / unit / restatement / parentheses confusion
pub const C_LAY: f64 = 0.006; // per layout difficulty level
pub const E_ARITH: f64 = 0.020; // per arithmetic operation
pub const RET_MISS_ERR: f64 = 0.70; // reader error when retrieval did not surface the needed span
// ---- costs (units of one full-document read)
pub const C_TOOL: f64 = 0.05;
pub const C_VER: f64 = 0.02;
pub const C_SPAN: f64 = 0.15;
pub const C_RET: f64 = 0.01;
// question -> tool-spec step (LLM reads only the question, not the document): error prob per agent multiplier and cost
pub const E_SPEC: f64 = 0.02;
pub const C_SPEC: f64 = 0.10;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VMode {
    Real,
    /// uninformative verifier (control): flags each kept fact w.p. p
    Random(u32),
    None,
}
#[derive(Clone, Debug)]
pub struct SimCfg {
    pub noise: f64,
    pub tau: f64,
    pub unseen_p: f64,
    pub n_per_world: usize,
    pub vc: VCfg,
    pub repair: bool,
    pub k_ret: usize,
    pub vmode: VMode,
    pub worlds: Vec<World>,
    pub layout_w: [f64; 5],
}
impl Default for SimCfg {
    fn default() -> Self {
        SimCfg { noise: 1.0, tau: 0.5, unseen_p: 0.4, n_per_world: 80, vc: VCfg::default(), repair: true, k_ret: 2, vmode: VMode::Real, worlds: vec![World::Paper, World::Fin, World::Biz], layout_w: [0.2, 0.2, 0.2, 0.25, 0.15] }
    }
}

#[derive(Clone, Debug)]
pub struct PS {
    pub nchars: usize,
    pub ndist: usize,
    pub m: f64,
    pub ops: f64,
    pub scale_conf: bool,
    pub layout: u8,
    pub qual: bool,
}
fn ops_of(op: QOp, nk: usize) -> f64 {
    match op {
        QOp::Ratio | QOp::Diff | QOp::Prod => 1.0,
        QOp::Growth => 2.0,
        QOp::EndDate => 3.0,
        QOp::ArgBest(_) => 0.5,
        QOp::Lookup | QOp::Qual => 0.0,
        QOp::CountSupported => nk as f64,
    }
}
fn ps_of(doc: &Doc, q: &Question, text: &str) -> PS {
    let nk = q.keys.len();
    let m = if q.op == QOp::CountSupported { 2.0 * nk as f64 } else { nk as f64 };
    let nd = count_numtok(text).saturating_sub(nk);
    PS { nchars: text.len(), ndist: nd, m, ops: ops_of(q.op, nk), scale_conf: doc.scale_conf, layout: doc.layout.min(4), qual: q.op == QOp::Qual }
}
pub fn p_fact_err(ps: &PS, agent: f64, noise: f64) -> f64 {
    (noise * (E0 * agent + A_DIS * ps.ndist.min(150) as f64 + B_LEN * (1.0 + ps.nchars as f64 / 800.0).ln() + C_SCALE * if ps.scale_conf { 1.0 } else { 0.0 } + C_LAY * ps.layout as f64)).clamp(0.0, 0.6)
}
pub fn p_plain_ok(ps: &PS, agent: f64, noise: f64) -> f64 {
    if ps.qual {
        return 1.0 - (noise * E0 * agent * 0.5).clamp(0.0, 0.5);
    }
    let p = p_fact_err(ps, agent, noise);
    (1.0 - p).powf(ps.m) * (1.0 - (noise * E_ARITH).clamp(0.0, 0.5)).powf(ps.ops)
}
pub fn p_span_ok(doc_scale_conf: bool, agent: f64, noise: f64) -> f64 {
    1.0 - (noise * (E0 * agent + C_SCALE * 0.5 * if doc_scale_conf { 1.0 } else { 0.0 })).clamp(0.0, 0.6)
}

// ---------------------------------------------------------------- retrieval
pub fn blocks_of(text: &str) -> Vec<(usize, usize)> {
    let mut v = Vec::new();
    let mut start: Option<usize> = None;
    let mut pos = 0;
    let mut last_end = 0;
    for l in text.split('\n') {
        if l.trim().is_empty() {
            if let Some(s) = start.take() {
                v.push((s, last_end));
            }
        } else {
            if start.is_none() {
                start = Some(pos);
            }
            last_end = pos + l.len();
        }
        pos += l.len() + 1;
    }
    if let Some(s) = start {
        v.push((s, last_end));
    }
    v
}
fn key_words(k: &str) -> String {
    let parts: Vec<&str> = k.split('|').collect();
    match parts[0] {
        "cell" => format!("{} {}", parts.get(1).unwrap_or(&""), METS.iter().find(|m| Some(&m.id) == parts.get(2)).map(|m| m.dev_name).unwrap_or("")),
        "clm" => "outperforms improves better than not accuracy latency [".to_string(),
        "kpi" => format!("{} {}", parts.get(1).unwrap_or(&""), parts.get(2).unwrap_or(&"")),
        "fee_monthly" => "fee pay per month".into(),
        "term_months" => "term months".into(),
        "notice_days" => "notice days terminate".into(),
        "cap" => "liability exceed".into(),
        "date" => "effective date entered into as of".into(),
        id => {
            let lab = super::gens::dev_labels().into_iter().find(|(i, _)| *i == id).map(|x| x.1).unwrap_or("");
            format!("{} {}", lab, parts.get(1).unwrap_or(&""))
        }
    }
}
/// returns (retrieved sub-text, all needed spans covered?)
pub fn retrieve(doc: &Doc, q: &Question, k: usize) -> (String, bool) {
    let bl = blocks_of(&doc.text);
    if bl.len() <= k + 1 {
        return (doc.text.clone(), true);
    }
    let strs: Vec<String> = bl.iter().map(|(a, b)| doc.text[*a..*b].to_string()).collect();
    let bm = Bm25::new(&strs);
    let mut query = q.text.clone();
    for key in &q.keys {
        query.push(' ');
        query.push_str(&key_words(key));
    }
    let mut sc = Vec::new();
    bm.score(&query, &mut sc);
    let mut order: Vec<usize> = (0..bl.len()).collect();
    order.sort_by(|&a, &b| sc[b].partial_cmp(&sc[a]).unwrap().then(a.cmp(&b)));
    let mut keep: Vec<usize> = vec![0];
    for &i in order.iter().take(k) {
        keep.push(i);
        if i > 0 {
            keep.push(i - 1);
        }
    }
    keep.sort();
    keep.dedup();
    let mut sub = String::new();
    for &i in &keep {
        sub.push_str(&strs[i]);
        sub.push_str("\n\n");
    }
    let covered = q.keys.iter().all(|key| match doc.fact(key) {
        Some(f) if f.hi > f.lo => keep.iter().any(|&i| bl[i].0 <= f.lo && f.hi <= bl[i].1),
        _ => true,
    });
    (sub, covered)
}

// ---------------------------------------------------------------- extraction outcome
#[derive(Clone, Debug)]
pub enum ExtOut {
    /// answer computed exactly from facts; ok = equals truth
    Exact { ok: bool, cost: f64 },
    /// facts missing -> full plain read of `ps`
    Full { cost: f64, ps: PS },
    /// flagged facts re-read (n spans); ok_if_clean = answer correct when all reads return truth
    Span { n: usize, ok_if_clean: bool, cost: f64, scale_conf: bool },
}
fn num_eq(op: QOp, a: f64, b: f64) -> bool {
    match op {
        QOp::ArgBest(_) | QOp::EndDate | QOp::CountSupported => a == b,
        _ => close(a, b),
    }
}
fn metric_lower(mid: &str) -> bool {
    METS.iter().find(|m| m.id == mid).map(|m| m.lower).unwrap_or(false)
}
pub fn ext_outcome(doc: &Doc, q: &Question, text: &str, ps_full: &PS, cfg: &SimCfg, verify: bool, seed: u64, idx: u64) -> ExtOut {
    let run = run_extract(doc.world, text, &ExCfg::default(), cfg.tau, &if verify && cfg.vmode == VMode::Real { cfg.vc } else { VCfg { span: false, scale: false, plaus: false, foot: false, xchk: false, cover: false, flag_ambiguous_foot: false, broad: false } });
    let kept: Vec<Ext> = run.kept.iter().map(|&i| run.all[i].clone()).collect();
    let mut cost = C_TOOL + if verify { C_VER } else { 0.0 };
    // verifier flags per kept fact
    let flag_of = |i: usize| -> (bool, Option<f64>) {
        if !verify {
            return (false, None);
        }
        match cfg.vmode {
            VMode::Real => (run.vfy[i].flagged(), run.vfy[i].repair),
            VMode::Random(p) => (R::new(seed, 555, idx, hash(i as u64, kept[i].fact.key.len() as u64, 0, 0)).f() < p as f64 / 1000.0, None),
            VMode::None => (false, None),
        }
    };
    let full = |cost: f64| ExtOut::Full { cost, ps: ps_full.clone() };
    if q.op == QOp::Qual {
        return full(cost);
    }
    if q.op == QOp::CountSupported {
        // claim support from extracted claims + cells
        let claims: Vec<(usize, &Ext)> = kept.iter().enumerate().filter(|(_, e)| e.fact.kind == FKind::Claim && e.fact.unit.starts_with("firm")).collect();
        let any_claim = kept.iter().any(|e| e.fact.kind == FKind::Claim);
        if !any_claim {
            return full(cost);
        }
        if verify && cfg.vmode == VMode::Real && cfg.vc.cover && coverage_gap(text, &kept) {
            return full(cost);
        }
        let (mut cnt, mut cnt_true) = (0.0, 0.0);
        let mut nflag = 0usize;
        for (i, c) in &claims {
            let parts: Vec<&str> = c.fact.key.split('|').collect();
            let (a, b, mid) = (parts[1], parts[2], parts[3]);
            let va = kept.iter().position(|e| e.fact.key == format!("cell|{}|{}", a, mid));
            let vb = kept.iter().position(|e| e.fact.key == format!("cell|{}|{}", b, mid));
            let (Some(va), Some(vb)) = (va, vb) else { return full(cost) };
            let better = if metric_lower(mid) { kept[va].fact.value < kept[vb].fact.value } else { kept[va].fact.value > kept[vb].fact.value };
            let sup = if c.fact.value > 0.0 { better } else { !better };
            cnt += if sup { 1.0 } else { 0.0 };
            // flagged inputs -> span reads
            let mut fl = flag_of(*i).0;
            fl |= flag_of(va).0 || flag_of(vb).0;
            if fl {
                nflag += 1;
            }
            // truth contribution of this extracted claim if reads were clean
            match doc.claims.iter().find(|t| t.key == c.fact.key && !t.hedged) {
                Some(t) => cnt_true += if t.supported { 1.0 } else { 0.0 },
                None => {}
            }
        }
        if nflag > 0 {
            // flagged inputs are re-read; after clean reads the count equals the count from truth-consistent claims
            let ok_clean = cnt_true == q.truth && claims.len() == doc.claims.iter().filter(|c| !c.hedged).count();
            return ExtOut::Span { n: nflag, ok_if_clean: ok_clean, cost: cost + C_SPAN * nflag as f64, scale_conf: doc.scale_conf };
        }
        return ExtOut::Exact { ok: cnt == q.truth, cost };
    }
    // numeric questions
    let mut vals = Vec::new();
    let mut vals_clean = Vec::new();
    let mut nflag = 0;
    for key in &q.keys {
        let Some(i) = kept.iter().position(|e| &e.fact.key == key) else { return full(cost) };
        let (fl, rep) = flag_of(i);
        let mut v = kept[i].fact.value;
        if fl {
            if cfg.repair && rep.is_some() {
                v = rep.unwrap();
                vals.push(v);
                vals_clean.push(v);
                continue;
            }
            nflag += 1;
            vals.push(v);
            vals_clean.push(doc.fact(key).map(|f| f.value).unwrap_or(v));
        } else {
            vals.push(v);
            vals_clean.push(v);
        }
    }
    let a = qeval(q.op, &vals);
    let a_clean = qeval(q.op, &vals_clean);
    if nflag > 0 {
        let ok = a_clean.map_or(false, |x| num_eq(q.op, x, q.truth));
        cost += C_SPAN * nflag as f64;
        return ExtOut::Span { n: nflag, ok_if_clean: ok, cost, scale_conf: doc.scale_conf };
    }
    match a {
        Some(x) => ExtOut::Exact { ok: num_eq(q.op, x, q.truth), cost },
        None => full(cost),
    }
}

#[derive(Clone, Debug)]
pub struct Prep {
    pub world: World,
    pub qop: QOp,
    pub ps: PS,
    pub ps_r: PS,
    pub ret_ok: bool,
    /// outcomes for modes E, E+R, EV, EV+R
    pub ext: [ExtOut; 4],
    pub scale_conf: bool,
}
pub const MODES: [&str; 6] = ["plain", "plain+R", "ext", "ext+R", "ext+ver", "ext+ver+R"];
pub fn prepare(seed: u64, idx: u64, cfg: &SimCfg, doc: Doc) -> Prep {
    let qi = (hash(seed, 17, idx, 0) % doc.qs.len().max(1) as u64) as usize;
    let q = doc.qs[qi].clone();
    let ps = ps_of(&doc, &q, &doc.text);
    let (sub, ret_ok) = retrieve(&doc, &q, cfg.k_ret);
    let ps_r = ps_of(&doc, &q, &sub);
    let e0 = ext_outcome(&doc, &q, &doc.text, &ps, cfg, false, seed, idx);
    let er = ext_outcome(&doc, &q, &sub, &ps_r, cfg, false, seed, idx);
    let ev = ext_outcome(&doc, &q, &doc.text, &ps, cfg, true, seed, idx);
    let evr = ext_outcome(&doc, &q, &sub, &ps_r, cfg, true, seed, idx);
    Prep { world: doc.world, qop: q.op, ps, ps_r, ret_ok, ext: [e0, er, ev, evr], scale_conf: doc.scale_conf }
}

// ---------------------------------------------------------------- routers
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RK {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Safe,
}
pub const RKS: [RK; 6] = [RK::Keyword, RK::Eps, RK::Pts, RK::HebR, RK::Warm, RK::Safe];
pub fn rk_name(k: RK) -> &'static str {
    match k {
        RK::Keyword => "keyword",
        RK::Eps => "eps-greedy",
        RK::Pts => "pooled-thompson",
        RK::HebR => "hebbian-robust",
        RK::Warm => "warm(hebr)",
        RK::Safe => "guarded-safe",
    }
}
pub fn mk_router(k: RK, kw: &HashMap<String, usize>) -> Box<dyn Router> {
    let n = NAG;
    match k {
        RK::Keyword => Box::new(KeywordRouter::new(kw.clone(), n)),
        RK::Eps => Box::new(EpsGreedyRouter::new(n, 0.97, 0.08)),
        RK::Pts => Box::new(PooledThompsonRouter::new(n, 0.97)),
        RK::HebR => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
        RK::Warm => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), kw.clone(), n, 8.0).with_evidence(Evidence::MeanFeature)),
        RK::Safe => Box::new(recommended_router(kw.clone(), n)),
    }
}

pub struct Session {
    /// agent reader multipliers [agent][world idx 0..3]
    pub mult: [[f64; 5]; NAG],
    pub kw: HashMap<String, usize>,
}
pub fn mk_session(seed: u64) -> Session {
    let mut mult = [[1.0; 5]; NAG];
    for a in 0..NAG {
        for w in 0..5 {
            mult[a][w] = 0.5 + 1.1 * R::new(seed, 901, a as u64, w as u64).f();
        }
    }
    let mut kw = HashMap::new();
    for w in [World::Paper, World::Fin, World::Biz, World::NoStruct, World::Tiny] {
        let best = (0..NAG).min_by(|&a, &b| mult[a][w.idx()].partial_cmp(&mult[b][w.idx()]).unwrap()).unwrap();
        // the keyword prior is right 60% of the time
        let pick = if R::new(seed, 902, w.idx() as u64, 0).f() < 0.6 { best } else { R::new(seed, 903, w.idx() as u64, 0).u(NAG) };
        kw.insert(w.name().to_string(), pick);
    }
    Session { mult, kw }
}
fn uni(seed: u64, idx: u64, tag: u64) -> f64 {
    R::new(seed, 700, idx, tag).f()
}

#[derive(Clone, Debug, Default)]
pub struct ModeRes {
    /// per world idx: (sum correct, sum cost, n)
    pub w: [(f64, f64, f64); 5],
}
impl ModeRes {
    pub fn acc(&self, w: Option<World>) -> f64 {
        let (s, _, n) = self.tot(w);
        if n > 0.0 { s / n } else { f64::NAN }
    }
    pub fn cost(&self, w: Option<World>) -> f64 {
        let (_, c, n) = self.tot(w);
        if n > 0.0 { c / n } else { f64::NAN }
    }
    fn tot(&self, w: Option<World>) -> (f64, f64, f64) {
        match w {
            Some(w) => self.w[w.idx()],
            None => self.w.iter().fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2)),
        }
    }
}

/// run one mode over the prepared session with router kind `rk`
pub fn run_mode(seed: u64, preps: &[Prep], sess: &Session, mode: usize, rk: RK, noise: f64) -> ModeRes {
    let mut router = mk_router(rk, &sess.kw);
    let mut res = ModeRes::default();
    let mut rng = neural_mesh::router::Rng::new(hash(seed, 31, mode as u64, rk as u64));
    for (t, p) in preps.iter().enumerate() {
        let idx = t as u64;
        let task = Task { id: idx, task_type: Some(p.world.name().to_string()), features: vec![p.world.name().to_string(), format!("{:?}", p.qop).chars().take(6).collect()] };
        let cands: Vec<usize> = (0..NAG).collect();
        let a = router.select(&task, &cands, &mut rng);
        let am = sess.mult[a][p.world.idx()];
        let (ok, cost) = match mode {
            0 => {
                let u = uni(seed, idx, 0);
                (u < p_plain_ok(&p.ps, am, noise), 1.0)
            }
            1 => {
                let u = uni(seed, idx, 0);
                let po = if p.ret_ok { p_plain_ok(&p.ps_r, am, noise) } else { 1.0 - RET_MISS_ERR };
                (u < po, 1.0 + C_RET)
            }
            _ => {
                let eo = &p.ext[mode - 2];
                let ret = mode == 3 || mode == 5;
                let base_c = if ret { C_RET } else { 0.0 };
                let spec_bad = uni(seed, idx, 1) < noise * E_SPEC * am;
                match eo {
                    ExtOut::Exact { ok, cost } => (*ok && !spec_bad, cost + base_c + C_SPEC),
                    ExtOut::Full { cost, ps } => {
                        let u = uni(seed, idx, 0);
                        let pps = if ret { &p.ps } else { ps };
                        // fallback after a failed retrieval/extraction reads the FULL document
                        (u < p_plain_ok(pps, am, noise), cost + 1.0 + base_c + C_SPEC)
                    }
                    ExtOut::Span { n, ok_if_clean, cost, scale_conf } => {
                        let mut all = *ok_if_clean;
                        for j in 0..*n {
                            if uni(seed, idx, 10 + j as u64) >= p_span_ok(*scale_conf, am, noise) {
                                all = false;
                            }
                        }
                        (all && !spec_bad, cost + base_c + C_SPEC)
                    }
                }
            }
        };
        let w = &mut res.w[p.world.idx()];
        w.0 += if ok { 1.0 } else { 0.0 };
        w.1 += cost;
        w.2 += 1.0;
        if uni(seed, idx, 99) < REVEAL {
            router.update(&task, a, if ok { 1.0 } else { 0.0 });
        }
    }
    res
}

/// generate + prepare the docs of one seed's session
pub fn prep_session(seed: u64, cfg: &SimCfg) -> Vec<Prep> {
    let mut v = Vec::new();
    let nw = cfg.worlds.len();
    for t in 0..cfg.n_per_world * nw {
        let w = cfg.worlds[t % nw];
        let doc = super::genb::gen_doc(w, seed, t as u64, cfg.unseen_p, &cfg.layout_w);
        v.push(prepare(seed, t as u64, cfg, doc));
    }
    v
}
