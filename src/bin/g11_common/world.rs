//! G11 worlds. WRITTEN AND FROZEN BEFORE ANY POLICY WAS RUN (only the unit tests below were run on them:
//! determinism, oracle > random). They do NOT share W7's generator (W7: 8 task types x 4 keywords, Bernoulli
//! reward, planted QUBO / max-coverage). Here: tasks carry a variable SET of requirement tags (+ irrelevant domain
//! tag + 30% random context tag), reward is a NOISY GRADED value clamp(v + 0.2 z, 0, 1) with a pre-drawn z per task
//! (common random numbers across policies). The primary metric is the NOISE-FREE expected team value v (policies
//! never see it); realized reward is reported too.
//!
//! Generators (each is a different team-value structure):
//!   RC    (+RC-D drift, +RC-L large) : requirement coverage. v = sum_s w_s * (1 - prod_{a in team}(1 - lev[a][s])),
//!         soft-OR coverage of weighted skill demands (submodular, redundancy gives diminishing returns).
//!         NOT in the learner's model class (no exact additive+pairwise form).
//!   ROLE  : role-based team. 3 of 5 roles needed (AND = product of per-role best quality), duplicate-role
//!         redundancy penalty. High-order interaction, NOT in the model class.
//!   THR   : capability thresholds. v = prod over 2 required dims of sigmoid((sum_a cap[a][d] - theta_d)/tau),
//!         additive capability inside a steep monotone link: NEAR the additive class (single-index model), not exactly in it.
//!   LAT   : latent-diversity team. v = 0.05 + 0.45 * mean_a q[type][a] + 0.5 * mean_{a<b} (1 - u_a.u_b)/2.
//!         EXACTLY additive + pairwise (own class BY CONSTRUCTION: positive control for the pairwise term, does
//!         NOT count as independent evidence).
//!   SEP, SEPT : separable CONTROLS (additive per-type skills; SEPT adds a trap keyword map). The learner's model
//!         class contains them; the question there is only speed of learning (harmlessness).
use neural_mesh::router::{Rng, Task};
use std::collections::HashMap;

pub static NOISE_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x3FC999999999999A); // 0.2
pub fn noise_sd() -> f64 {
    f64::from_bits(NOISE_BITS.load(std::sync::atomic::Ordering::Relaxed))
}
pub fn set_noise(x: f64) {
    NOISE_BITS.store(x.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WKind {
    Rc,
    Role,
    Thr,
    Lat,
    Sep,
    Comp,
}

#[derive(Clone, Debug)]
pub struct Cfg {
    pub name: &'static str,
    /// generator family (independence is counted per generator)
    pub family: &'static str,
    pub kind: WKind,
    pub n: usize,
    pub k: usize,
    pub t: usize,
    /// skills (Rc), roles (Role), dims (Thr), types (Lat, Sep)
    pub s: usize,
    pub drift: bool,
    pub trap: bool,
    pub control: bool,
    /// true when the value function is exactly additive+pairwise (does not count as independent evidence)
    pub own_class: bool,
    /// keyword map degraded: each tag's "obvious lead" is replaced by a random agent with prob 0.5 (stale/wrong routing table)
    pub kwbad: bool,
    /// 1 = worlds frozen before any policy ran; 2 = keyword-degraded variants ADDED AFTER the dev run showed keyword ~ oracle in set 1
    pub set: u8,
}

pub fn worlds() -> Vec<Cfg> {
    let b = Cfg { name: "", family: "", kind: WKind::Rc, n: 14, k: 4, t: 2500, s: 8, drift: false, trap: false, control: false, own_class: false, kwbad: false, set: 1 };
    vec![
        Cfg { name: "RC", family: "RC", kind: WKind::Rc, ..b.clone() },
        Cfg { name: "RC-drift", family: "RC", kind: WKind::Rc, drift: true, ..b.clone() },
        Cfg { name: "RC-L", family: "RC", kind: WKind::Rc, n: 24, k: 5, t: 3500, s: 10, ..b.clone() },
        Cfg { name: "ROLE", family: "ROLE", kind: WKind::Role, n: 15, k: 4, s: 5, ..b.clone() },
        Cfg { name: "THR", family: "THR", kind: WKind::Thr, n: 14, k: 4, s: 3, ..b.clone() },
        Cfg { name: "LAT", family: "LAT", kind: WKind::Lat, n: 12, k: 4, s: 6, own_class: true, ..b.clone() },
        Cfg { name: "C-SEP", family: "SEP", kind: WKind::Sep, n: 10, k: 3, s: 8, control: true, ..b.clone() },
        Cfg { name: "C-SEPT", family: "SEP", kind: WKind::Sep, n: 16, k: 4, s: 8, trap: true, control: true, ..b.clone() },
        Cfg { name: "RC-K", family: "RC", kind: WKind::Rc, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "RC-drift-K", family: "RC", kind: WKind::Rc, drift: true, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "RC-L-K", family: "RC", kind: WKind::Rc, n: 24, k: 5, t: 3500, s: 10, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "ROLE-K", family: "ROLE", kind: WKind::Role, n: 15, k: 4, s: 5, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "THR-K", family: "THR", kind: WKind::Thr, n: 14, k: 4, s: 3, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "LAT-K", family: "LAT", kind: WKind::Lat, n: 12, k: 4, s: 6, own_class: true, kwbad: true, set: 2, ..b.clone() },
        Cfg { name: "COMP", family: "COMP", kind: WKind::Comp, n: 10, k: 3, s: 8, set: 3, ..b.clone() },
        Cfg { name: "COMP-X", family: "COMPX", kind: WKind::Comp, n: 9, k: 3, s: 8, trap: true, set: 3, ..b.clone() },
    ]
}

pub fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
}

fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct World {
    pub cfg: Cfg,
    pub tasks: Vec<Task>,
    pub z: Vec<f64>,
    pub kw: HashMap<String, usize>,
    lev: Vec<f64>,      // Rc: n*s skill levels ; Thr: n*s capabilities ; Role: n (quality)
    lev_post: Vec<f64>, // Rc drift
    role: Vec<usize>,   // Role: role of each agent
    need: Vec<Vec<usize>>,
    wt: Vec<Vec<f64>>, // Rc weights / Thr thresholds
    ty: Vec<usize>,    // Lat, Sep task type
    sep: Vec<f64>,     // Sep: s[ty*n+a] ; Lat: q[ty*n+a]
    pairs: Vec<(usize, usize)>, // Comp: hidden complementary pair per task type
    uv: Vec<[f64; 3]>,
    drift_at: usize,
}

impl World {
    pub fn new(seed: u64, cfg: Cfg) -> World {
        let mut hr = Rng::new(seed ^ 0x6611_AA55_0F0F_1234);
        let mut tr = Rng::new(seed ^ 0x1B1B_C3C3_9999_7777);
        let (n, s, t) = (cfg.n, cfg.s, cfg.t);
        let mut w = World {
            tasks: Vec::with_capacity(t),
            z: Vec::with_capacity(t),
            kw: HashMap::new(),
            lev: vec![],
            lev_post: vec![],
            role: vec![],
            need: vec![],
            wt: vec![],
            ty: vec![],
            sep: vec![],
            uv: vec![],
            pairs: vec![],
            drift_at: t / 2,
            cfg: cfg.clone(),
        };
        match cfg.kind {
            WKind::Rc => {
                w.lev = Self::gen_rc_levels(n, s, &mut hr, true);
                if cfg.drift {
                    let mut lp = w.lev.clone();
                    let mut ag: Vec<usize> = (0..n).collect();
                    shuffle(&mut ag, &mut hr);
                    let fresh = Self::gen_rc_levels(n, s, &mut hr, false);
                    for &a in &ag[..n / 3] {
                        for j in 0..s {
                            lp[a * s + j] = fresh[a * s + j];
                        }
                    }
                    w.lev_post = lp;
                } else {
                    w.lev_post = w.lev.clone();
                }
                for j in 0..s {
                    w.kw.insert(format!("need:s{j}"), argmax((0..n).map(|a| w.lev[a * s + j])));
                }
            }
            WKind::Role => {
                let mut ag: Vec<usize> = (0..n).collect();
                shuffle(&mut ag, &mut hr);
                w.role = vec![0; n];
                for (i, &a) in ag.iter().enumerate() {
                    w.role[a] = i % s;
                }
                w.lev = (0..n).map(|_| 0.35 + 0.65 * hr.f64()).collect();
                for r in 0..s {
                    let lead = argmax((0..n).map(|a| if w.role[a] == r { w.lev[a] } else { -1.0 }));
                    w.kw.insert(format!("needs:r{r}"), lead);
                }
            }
            WKind::Thr => {
                w.lev = vec![0.0; n * s];
                let mut ag: Vec<usize> = (0..n).collect();
                shuffle(&mut ag, &mut hr);
                for (i, &a) in ag.iter().enumerate() {
                    let main = i % s;
                    for d in 0..s {
                        w.lev[a * s + d] = if d == main { 0.5 + 0.5 * hr.f64() } else { 0.25 * hr.f64() };
                    }
                }
                for d in 0..s {
                    w.kw.insert(format!("dim:d{d}"), argmax((0..n).map(|a| w.lev[a * s + d])));
                }
            }
            WKind::Lat => {
                w.sep = (0..s * n).map(|_| hr.f64()).collect();
                for _ in 0..n {
                    loop {
                        let v = [normal(&mut hr), normal(&mut hr), normal(&mut hr)];
                        let nr = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                        if nr > 1e-6 {
                            w.uv.push([v[0] / nr, v[1] / nr, v[2] / nr]);
                            break;
                        }
                    }
                }
                for ty in 0..s {
                    w.kw.insert(format!("kind:{ty}"), argmax((0..n).map(|a| w.sep[ty * n + a])));
                }
            }
            WKind::Comp => {
                w.sep = (0..s * n).map(|_| 0.08 * hr.f64()).collect();
                if cfg.trap {
                    // COMP-X: only 3 hidden pairs; type ty rewards pair (ty mod 3) and PENALISES the other two pairs (task-conditioned interaction)
                    let mut ag: Vec<usize> = (0..n).collect();
                    shuffle(&mut ag, &mut hr);
                    for p in 0..3 {
                        w.pairs.push((ag[2 * p], ag[2 * p + 1]));
                    }
                    for ty in 0..s {
                        w.kw.insert(format!("kind:{ty}"), w.pairs[ty % 3].0);
                    }
                } else {
                    for ty in 0..s {
                        let mut ag: Vec<usize> = (0..n).collect();
                        shuffle(&mut ag, &mut hr);
                        w.pairs.push((ag[0], ag[1]));
                        w.kw.insert(format!("kind:{ty}"), ag[0]);
                    }
                }
            }
            WKind::Sep => {
                w.sep = vec![0.0; s * n];
                let ntop = cfg.k;
                let mut tops = vec![0usize; s];
                for ty in 0..s {
                    let mut ag: Vec<usize> = (0..n).collect();
                    shuffle(&mut ag, &mut hr);
                    for (r, &a) in ag.iter().enumerate() {
                        w.sep[ty * n + a] = if r < ntop { 0.12 + 0.1 * hr.f64() } else { 0.03 * hr.f64() };
                    }
                    tops[ty] = argmax((0..n).map(|a| w.sep[ty * n + a]));
                }
                for ty in 0..s {
                    let tgt = if cfg.trap && ty >= s / 2 { argmin((0..n).map(|a| w.sep[ty * n + a])) } else { tops[ty] };
                    w.kw.insert(format!("kind:{ty}"), tgt);
                }
            }
        }
        if cfg.kwbad {
            let mut kr = Rng::new(seed ^ 0x7777_BAD0_1111_2222);
            let mut keys: Vec<String> = w.kw.keys().cloned().collect();
            keys.sort();
            for key in keys {
                if kr.f64() < 0.5 {
                    w.kw.insert(key, kr.below(n));
                }
            }
        }
        // tasks
        for id in 0..t {
            let mut feats: Vec<String> = vec![];
            match cfg.kind {
                WKind::Rc => {
                    let nreq = 2 + tr.below(3);
                    let mut sk: Vec<usize> = (0..s).collect();
                    shuffle(&mut sk, &mut tr);
                    let req = sk[..nreq].to_vec();
                    let mut wts: Vec<f64> = (0..nreq).map(|_| 0.5 + tr.f64()).collect();
                    let sm: f64 = wts.iter().sum();
                    wts.iter_mut().for_each(|x| *x /= sm);
                    for &j in &req {
                        feats.push(format!("need:s{j}"));
                    }
                    w.need.push(req);
                    w.wt.push(wts);
                }
                WKind::Role => {
                    let mut rr: Vec<usize> = (0..s).collect();
                    shuffle(&mut rr, &mut tr);
                    let req = rr[..3].to_vec();
                    for &r in &req {
                        feats.push(format!("needs:r{r}"));
                    }
                    w.need.push(req);
                    w.wt.push(vec![]);
                }
                WKind::Thr => {
                    let mut dd: Vec<usize> = (0..s).collect();
                    shuffle(&mut dd, &mut tr);
                    let req = dd[..2].to_vec();
                    let th: Vec<f64> = req.iter().map(|_| 0.8 + 0.7 * tr.f64()).collect();
                    for (i, &d) in req.iter().enumerate() {
                        feats.push(format!("dim:d{d}"));
                        if th[i] > 1.15 {
                            feats.push(format!("hard:d{d}"));
                        }
                    }
                    w.need.push(req);
                    w.wt.push(th);
                }
                WKind::Lat | WKind::Sep | WKind::Comp => {
                    let ty = tr.below(s);
                    feats.push(format!("kind:{ty}"));
                    w.ty.push(ty);
                    w.need.push(vec![]);
                    w.wt.push(vec![]);
                }
            }
            feats.push(format!("dom:{}", tr.below(3)));
            if tr.f64() < 0.3 {
                feats.push(format!("ctx:{}", tr.below(6)));
            }
            feats.sort();
            w.tasks.push(Task { id: id as u64, task_type: None, features: feats });
            w.z.push(normal(&mut tr));
        }
        w
    }

    fn gen_rc_levels(n: usize, s: usize, rng: &mut Rng, ensure: bool) -> Vec<f64> {
        loop {
            let mut lev = vec![0.0; n * s];
            for a in 0..n {
                for j in 0..s {
                    lev[a * s + j] = 0.12 * rng.f64();
                }
                let nm = 1 + (rng.f64() < 0.5) as usize;
                let mut sk: Vec<usize> = (0..s).collect();
                shuffle(&mut sk, rng);
                for &j in &sk[..nm] {
                    lev[a * s + j] = 0.6 + 0.35 * rng.f64();
                }
            }
            if !ensure || (0..s).all(|j| (0..n).filter(|&a| lev[a * s + j] >= 0.6).count() >= 2) {
                return lev;
            }
        }
    }

    /// noise-free expected team value in [0,1]
    pub fn value(&self, t: usize, team: &[usize]) -> f64 {
        let (n, s) = (self.cfg.n, self.cfg.s);
        let v = match self.cfg.kind {
            WKind::Rc => {
                let lev = if t >= self.drift_at { &self.lev_post } else { &self.lev };
                let mut v = 0.0;
                for (i, &j) in self.need[t].iter().enumerate() {
                    let miss: f64 = team.iter().map(|&a| 1.0 - lev[a * s + j]).product();
                    v += self.wt[t][i] * (1.0 - miss);
                }
                v
            }
            WKind::Role => {
                let mut v = 1.0;
                for &r in &self.need[t] {
                    let cov = team.iter().map(|&a| if self.role[a] == r { self.lev[a] } else { 0.3 * self.lev[a] }).fold(0.0, f64::max);
                    v *= 0.1 + 0.9 * cov;
                }
                let mut seen = vec![false; s];
                let mut distinct = 0usize;
                for &a in team {
                    if !seen[self.role[a]] {
                        seen[self.role[a]] = true;
                        distinct += 1;
                    }
                }
                v * (1.0 - 0.06 * (team.len() - distinct) as f64)
            }
            WKind::Thr => {
                let mut v = 1.0;
                for (i, &d) in self.need[t].iter().enumerate() {
                    let tot: f64 = team.iter().map(|&a| self.lev[a * s + d]).sum();
                    v *= sigmoid((tot - self.wt[t][i]) / 0.15);
                }
                v
            }
            WKind::Lat => {
                let ty = self.ty[t];
                let qn = team.iter().map(|&a| self.sep[ty * n + a]).sum::<f64>() / team.len() as f64;
                let mut dn = 0.0;
                let mut np = 0.0;
                for (x, &a) in team.iter().enumerate() {
                    for &b in &team[x + 1..] {
                        let d = self.uv[a][0] * self.uv[b][0] + self.uv[a][1] * self.uv[b][1] + self.uv[a][2] * self.uv[b][2];
                        dn += (1.0 - d) / 2.0;
                        np += 1.0;
                    }
                }
                0.05 + 0.45 * qn + 0.5 * dn / f64::max(np, 1.0)
            }
            WKind::Comp => {
                let ty = self.ty[t];
                let syn = if self.cfg.trap {
                    let mut sy = 0.0;
                    for (pi, &(pa, pb)) in self.pairs.iter().enumerate() {
                        if team.contains(&pa) && team.contains(&pb) {
                            sy += if pi == ty % 3 { 0.45 } else { -0.25 };
                        }
                    }
                    sy
                } else {
                    let (pa, pb) = self.pairs[ty];
                    if team.contains(&pa) && team.contains(&pb) { 0.45 } else { 0.0 }
                };
                0.1 + syn + team.iter().map(|&a| self.sep[ty * n + a]).sum::<f64>()
            }
            WKind::Sep => {
                let ty = self.ty[t];
                0.1 + team.iter().map(|&a| self.sep[ty * n + a]).sum::<f64>()
            }
        };
        v.clamp(0.0, 1.0)
    }

    pub fn reward(&self, t: usize, team: &[usize]) -> f64 {
        (self.value(t, team) + noise_sd() * self.z[t]).clamp(0.0, 1.0)
    }

    /// (oracle mean v, random-team mean v), estimated on <= 120 evenly spaced tasks (oracle: exhaustive over all teams).
    pub fn references(&self, seed: u64) -> (f64, f64) {
        let (n, k) = (self.cfg.n, self.cfg.k);
        let step = (self.cfg.t / 120).max(1);
        let mut rng = Rng::new(seed ^ 0x5151);
        let (mut o, mut r, mut c) = (0.0, 0.0, 0.0);
        let mut t = 0;
        while t < self.cfg.t {
            let mut b = 0.0f64;
            combos(n, k, |tm| {
                b = b.max(self.value(t, tm));
            });
            o += b;
            let mut sm = 0.0;
            for _ in 0..30 {
                let mut ag: Vec<usize> = (0..n).collect();
                shuffle(&mut ag, &mut rng);
                sm += self.value(t, &ag[..k]);
            }
            r += sm / 30.0;
            c += 1.0;
            t += step;
        }
        (o / c, r / c)
    }
}

fn argmax(it: impl Iterator<Item = f64>) -> usize {
    let mut bi = 0;
    let mut bv = f64::NEG_INFINITY;
    for (i, v) in it.enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi
}
fn argmin(it: impl Iterator<Item = f64>) -> usize {
    argmax(it.map(|x| -x))
}

pub fn combos<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
    if k == 0 || k > n {
        return;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        f(&idx);
        let mut i = k;
        loop {
            if i == 0 {
                return;
            }
            i -= 1;
            if idx[i] != i + n - k {
                break;
            }
            if i == 0 {
                return;
            }
        }
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_oracle_beats_random() {
        for cfg in worlds() {
            let a = World::new(3, cfg.clone());
            let b = World::new(3, cfg.clone());
            assert_eq!(a.tasks, b.tasks);
            let team: Vec<usize> = (0..cfg.k).collect();
            assert_eq!(a.value(10, &team), b.value(10, &team));
            assert!(a.kw.values().all(|&x| x < cfg.n));
            if cfg.n <= 16 {
                let (o, r) = a.references(3);
                assert!(o > r + 0.1, "{} oracle {o} random {r}", cfg.name);
            }
            let v = a.value(5, &team);
            assert!((0.0..=1.0).contains(&v));
        }
    }
}

#[cfg(test)]
mod tests_noise {
    #[test]
    fn default_noise_is_point_two() {
        assert_eq!(super::noise_sd(), 0.2);
    }
}
