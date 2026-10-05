//! G7 worlds. DEFINED BEFORE ANY POLICY WAS RUN; not edited after (bug fixes noted in the board entry).
//!
//! Part A (agent TEAM assignment): a team of exactly K agents is chosen per task (binary vector x, |x|=K).
//!   * Sep   : additive per-type skills, no interaction. TRIVIALLY SEPARABLE control (optimum = top-K singles).
//!   * Cover : complementary skills. Each type needs 3 of 6 skills, each agent holds 2 skills; success depends
//!             on COVERAGE of the needed skills by the team union (redundant agents add nothing).
//!   * Syn   : weak singles + hidden global PAIRWISE synergy (planted QUBO); other pairs carry a coordination cost.
//!   * SynDrift: Syn, but half of the synergy pairs change at T/2.
//!   * SynL  : Syn with N=30, K=6 (C(30,6)=593775 teams: exhaustive search per decision is not cheap).
//! Part B (FEATURE -> AGENT assignment with congestion): binary one-hot matrix Z[f][a]. An agent that received
//!   more than `cap` of the last `L` tasks has its success probability multiplied by max(floor, 1-kappa*(load-cap)).
//!   * BSep  : kappa = 0 (separable control: optimum = per-feature argmax).
//!   * BCong : congestion; the top agents of 6 of 8 types sit on two "hot" agents.
//!   * BCongDrift: BCong + two types change their best agent at T/2.
//! Reward is Bernoulli(p) driven by one pre-drawn uniform per task (common random numbers across policies).
use neural_mesh::router::{Rng, Task};
use std::collections::HashMap;

pub const NTYPES: usize = 8;
const NKW: usize = 4;

fn kw_name(ty: usize, j: usize) -> String {
    format!("k{ty}_{j}")
}

pub fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
}

/// Shared task stream: type, 2-3 of its 4 keywords, 25% chance of a confound keyword from another type.
fn gen_tasks(seed: u64, t: usize) -> (Vec<Task>, Vec<usize>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut tasks = Vec::with_capacity(t);
    let mut types = Vec::with_capacity(t);
    let mut us = Vec::with_capacity(t);
    for id in 0..t {
        let ty = rng.below(NTYPES);
        let nk = 2 + rng.below(2);
        let mut ks: Vec<usize> = (0..NKW).collect();
        shuffle(&mut ks, &mut rng);
        let mut feats: Vec<String> = ks[..nk].iter().map(|&j| kw_name(ty, j)).collect();
        if rng.f64() < 0.25 {
            let oty = (ty + 1 + rng.below(NTYPES - 1)) % NTYPES;
            feats.push(kw_name(oty, rng.below(NKW)));
        }
        feats.sort();
        tasks.push(Task { id: id as u64, task_type: None, features: feats });
        types.push(ty);
        us.push(rng.f64());
    }
    (tasks, types, us)
}

/// Dense feature index used by the optimizer arms: "k{ty}_{j}" -> ty*4+j.
pub fn feat_index(name: &str) -> usize {
    let r = &name[1..];
    let mut it = r.split('_');
    let ty: usize = it.next().unwrap().parse().unwrap();
    let j: usize = it.next().unwrap().parse().unwrap();
    ty * NKW + j
}
pub const NFEAT: usize = NTYPES * NKW;

// ============================================================ Part A
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeamKind {
    Sep,
    Cover,
    Syn,
}

#[derive(Clone, Debug)]
pub struct TeamCfg {
    pub name: &'static str,
    pub kind: TeamKind,
    pub n: usize,
    pub k: usize,
    pub t: usize,
    pub drift: bool,
    pub n_pairs: usize,
    /// Cover only: number of skills and skills required per type
    pub n_skills: usize,
    pub n_req: usize,
}

pub fn team_worlds() -> Vec<TeamCfg> {
    vec![
        TeamCfg { name: "A-sep", kind: TeamKind::Sep, n: 8, k: 3, t: 1500, drift: false, n_pairs: 0, n_skills: 0, n_req: 0 },
        TeamCfg { name: "A-cover", kind: TeamKind::Cover, n: 10, k: 3, t: 1500, drift: false, n_pairs: 0, n_skills: 6, n_req: 3 },
        TeamCfg { name: "A-cover-L", kind: TeamKind::Cover, n: 24, k: 5, t: 3000, drift: false, n_pairs: 0, n_skills: 10, n_req: 4 },
        TeamCfg { name: "A-syn", kind: TeamKind::Syn, n: 12, k: 4, t: 2000, drift: false, n_pairs: 4, n_skills: 0, n_req: 0 },
        TeamCfg { name: "A-syn-drift", kind: TeamKind::Syn, n: 12, k: 4, t: 2000, drift: true, n_pairs: 4, n_skills: 0, n_req: 0 },
        TeamCfg { name: "A-syn-L", kind: TeamKind::Syn, n: 30, k: 6, t: 3000, drift: false, n_pairs: 8, n_skills: 0, n_req: 0 },
    ]
}

pub struct TeamWorld {
    pub cfg: TeamCfg,
    pub tasks: Vec<Task>,
    pub ttype: Vec<usize>,
    pub u: Vec<f64>,
    /// keyword -> obvious lead agent
    pub kw: HashMap<String, usize>,
    s: Vec<Vec<f64>>,     // [type][agent] singles (Sep, Syn)
    prof: Vec<Vec<f64>>,  // [agent][skill] (Cover)
    req: Vec<Vec<usize>>, // [type] required skills (Cover)
    j_pre: Vec<f64>,      // n*n pair term (Syn)
    j_post: Vec<f64>,
    drift_at: usize,
    base: f64,
}

pub const P_MIN: f64 = 0.02;
pub const P_MAX: f64 = 0.97;
const SYN_PLUS: f64 = 0.28;
const SYN_COST: f64 = -0.03;

impl TeamWorld {
    pub fn new(seed: u64, cfg: TeamCfg) -> Self {
        let (tasks, ttype, u) = gen_tasks(seed, cfg.t);
        let mut hr = Rng::new(seed ^ 0xA7A7_1234_5678_9ABC);
        let n = cfg.n;
        let mut s = vec![vec![0.0; n]; NTYPES];
        let ns = cfg.n_skills.max(1);
        let mut prof = vec![vec![0.0; ns]; n];
        let mut req = vec![vec![]; NTYPES];
        let mut jpre = vec![0.0; n * n];
        let mut jpost = vec![0.0; n * n];
        let mut lead = vec![0usize; NTYPES];
        let base;
        match cfg.kind {
            TeamKind::Sep => {
                base = 0.08;
                for ty in 0..NTYPES {
                    let mut ag: Vec<usize> = (0..n).collect();
                    shuffle(&mut ag, &mut hr);
                    for (r, &a) in ag.iter().enumerate() {
                        s[ty][a] = if r < 3 { 0.22 + 0.08 * hr.f64() } else { 0.04 * hr.f64() };
                    }
                    lead[ty] = ag[0];
                }
            }
            TeamKind::Cover => {
                base = 0.05;
                loop {
                    for a in 0..n {
                        prof[a] = vec![0.0; ns];
                        let mut sk: Vec<usize> = (0..ns).collect();
                        shuffle(&mut sk, &mut hr);
                        for &k in &sk[..2] {
                            prof[a][k] = 0.7 + 0.3 * hr.f64();
                        }
                    }
                    if (0..ns).all(|k| (0..n).filter(|&a| prof[a][k] > 0.0).count() >= 2) {
                        break;
                    }
                }
                for ty in 0..NTYPES {
                    let mut sk: Vec<usize> = (0..ns).collect();
                    shuffle(&mut sk, &mut hr);
                    req[ty] = sk[..cfg.n_req].to_vec();
                    let mut best = (0usize, -1.0);
                    for a in 0..n {
                        let v: f64 = req[ty].iter().map(|&k| prof[a][k]).sum();
                        if v > best.1 {
                            best = (a, v);
                        }
                    }
                    lead[ty] = best.0;
                }
            }
            TeamKind::Syn => {
                base = 0.15;
                for ty in 0..NTYPES {
                    for a in 0..n {
                        s[ty][a] = 0.12 * hr.f64();
                    }
                    lead[ty] = hr.below(n);
                    s[ty][lead[ty]] = 0.2;
                }
                for i in 0..n {
                    for j in 0..n {
                        if i != j {
                            jpre[i * n + j] = SYN_COST;
                            jpost[i * n + j] = SYN_COST;
                        }
                    }
                }
                let mut ag: Vec<usize> = (0..n).collect();
                shuffle(&mut ag, &mut hr);
                let np = cfg.n_pairs;
                let set_pair = |m: &mut Vec<f64>, a: usize, b: usize| {
                    m[a * n + b] = SYN_PLUS;
                    m[b * n + a] = SYN_PLUS;
                };
                for p in 0..np {
                    set_pair(&mut jpre, ag[2 * p], ag[2 * p + 1]);
                }
                if cfg.drift {
                    // pairs 0,1 are replaced by two new pairs among never-used agents; pairs 2.. persist
                    for p in 2..np {
                        set_pair(&mut jpost, ag[2 * p], ag[2 * p + 1]);
                    }
                    let free = &ag[2 * np..];
                    assert!(free.len() >= 4, "drift needs 4 unused agents");
                    set_pair(&mut jpost, free[0], free[1]);
                    set_pair(&mut jpost, free[2], free[3]);
                } else {
                    jpost = jpre.clone();
                }
            }
        }
        if cfg.kind != TeamKind::Syn {
            jpost = jpre.clone();
        }
        let mut kw = HashMap::new();
        for ty in 0..NTYPES {
            for j in 0..NKW {
                kw.insert(kw_name(ty, j), lead[ty]);
            }
        }
        TeamWorld { drift_at: cfg.t / 2, cfg, tasks, ttype, u, kw, s, prof, req, j_pre: jpre, j_post: jpost, base }
    }

    pub fn prob_at(&self, t: usize, team: &[usize]) -> f64 {
        let ty = self.ttype[t];
        let post = self.cfg.drift && t >= self.drift_at;
        self.prob_ty(ty, post, team)
    }

    fn prob_ty(&self, ty: usize, post: bool, team: &[usize]) -> f64 {
        let n = self.cfg.n;
        let p = match self.cfg.kind {
            TeamKind::Sep => self.base + team.iter().map(|&a| self.s[ty][a]).sum::<f64>(),
            TeamKind::Cover => {
                let cov: f64 = self.req[ty].iter().map(|&k| team.iter().map(|&a| self.prof[a][k]).fold(0.0, f64::max)).sum::<f64>() / self.cfg.n_req as f64;
                self.base + 0.9 * cov
            }
            TeamKind::Syn => {
                let j = if post { &self.j_post } else { &self.j_pre };
                let mut v = self.base + team.iter().map(|&a| self.s[ty][a]).sum::<f64>();
                for (x, &a) in team.iter().enumerate() {
                    for &b in &team[x + 1..] {
                        v += j[a * n + b];
                    }
                }
                v
            }
        };
        p.clamp(P_MIN, P_MAX)
    }

    pub fn reward(&self, t: usize, team: &[usize]) -> f64 {
        if self.u[t] < self.prob_at(t, team) { 1.0 } else { 0.0 }
    }

    /// (oracle mean p, random-team mean p) over the task stream. oracle = NaN when C(n,k) > 30000.
    pub fn references(&self, seed: u64) -> (f64, f64) {
        let (n, k) = (self.cfg.n, self.cfg.k);
        let mut cnt = 1.0f64;
        for i in 0..k {
            cnt = cnt * (n - i) as f64 / (i + 1) as f64;
        }
        let mut best = vec![[f64::NAN; 2]; NTYPES];
        if cnt <= 30000.0 {
            for ty in 0..NTYPES {
                for post in 0..2 {
                    let mut b = 0.0f64;
                    for_each_combo(n, k, |c| {
                        b = b.max(self.prob_ty(ty, post == 1, c));
                    });
                    best[ty][post] = b;
                }
            }
        }
        let mut rng = Rng::new(seed ^ 0x5151);
        let mut rnd = vec![[0.0f64; 2]; NTYPES];
        for ty in 0..NTYPES {
            for post in 0..2 {
                let mut sm = 0.0;
                for _ in 0..200 {
                    let mut ag: Vec<usize> = (0..n).collect();
                    shuffle(&mut ag, &mut rng);
                    sm += self.prob_ty(ty, post == 1, &ag[..k]);
                }
                rnd[ty][post] = sm / 200.0;
            }
        }
        let (mut o, mut r) = (0.0, 0.0);
        for t in 0..self.cfg.t {
            let post = (self.cfg.drift && t >= self.drift_at) as usize;
            o += best[self.ttype[t]][post];
            r += rnd[self.ttype[t]][post];
        }
        (o / self.cfg.t as f64, r / self.cfg.t as f64)
    }
}

/// Visit every k-subset of 0..n (as a sorted slice).
pub fn for_each_combo<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
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

// ============================================================ Part B
#[derive(Clone, Debug)]
pub struct AsgCfg {
    pub name: &'static str,
    pub t: usize,
    pub kappa: f64,
    pub drift: bool,
}

pub const B_AGENTS: usize = 8;
pub const B_CAP: usize = 6;
pub const B_WIN: usize = 20;
pub const B_FLOOR: f64 = 0.15;

pub fn asg_worlds() -> Vec<AsgCfg> {
    vec![
        AsgCfg { name: "B-sep", t: 1500, kappa: 0.0, drift: false },
        AsgCfg { name: "B-cong", t: 1500, kappa: 0.15, drift: false },
        AsgCfg { name: "B-cong-drift", t: 1500, kappa: 0.15, drift: true },
    ]
}

pub struct AsgWorld {
    pub cfg: AsgCfg,
    pub tasks: Vec<Task>,
    pub ttype: Vec<usize>,
    pub u: Vec<f64>,
    pub kw: HashMap<String, usize>,
    q_pre: Vec<Vec<f64>>,
    q_post: Vec<Vec<f64>>,
    drift_at: usize,
}

impl AsgWorld {
    pub fn new(seed: u64, cfg: AsgCfg) -> Self {
        let (tasks, ttype, u) = gen_tasks(seed, cfg.t);
        let mut hr = Rng::new(seed ^ 0xB7B7_4321_8765_CBA9);
        let a = B_AGENTS;
        // random role permutation: first two = hot agents; the rest = cold agents
        let mut perm: Vec<usize> = (0..a).collect();
        shuffle(&mut perm, &mut hr);
        let (h1, h2) = (perm[0], perm[1]);
        let cold = &perm[2..];
        let mut q = vec![vec![0.0; a]; NTYPES];
        for ty in 0..NTYPES {
            for ag in 0..a {
                q[ty][ag] = 0.15 + 0.35 * hr.f64();
            }
        }
        // top agents: types 0,1,2 -> h1 ; 3,4,5 -> h2 ; 6 -> cold[0] ; 7 -> cold[1].
        let top = [h1, h1, h1, h2, h2, h2, cold[0], cold[1]];
        for ty in 0..NTYPES {
            q[ty][top[ty]] = 0.85;
        }
        // good alternatives so that rebalancing is possible (cheap for types 2,5; costlier for 0,1,3,4)
        q[2][cold[2]] = 0.70;
        q[5][cold[3]] = 0.70;
        q[0][cold[4]] = 0.62;
        q[1][cold[5]] = 0.62;
        q[3][cold[2]] = 0.62;
        q[4][cold[3]] = 0.62;
        let mut q_post = q.clone();
        if cfg.drift {
            // types 1 and 4 get a new best (a cold agent) and the old best collapses
            q_post[1][top[1]] = 0.35;
            q_post[1][cold[0]] = 0.85;
            q_post[4][top[4]] = 0.35;
            q_post[4][cold[1]] = 0.85;
        }
        // keyword map: correct for types 0..5; traps for 6 and 7 (map to the worst agent)
        let mut kw = HashMap::new();
        for ty in 0..NTYPES {
            let tgt = if ty >= 6 { (0..a).min_by(|&x, &y| q[ty][x].total_cmp(&q[ty][y])).unwrap() } else { top[ty] };
            for j in 0..NKW {
                kw.insert(kw_name(ty, j), tgt);
            }
        }
        AsgWorld { drift_at: cfg.t / 2, cfg, tasks, ttype, u, kw, q_pre: q, q_post }
    }

    /// success probability of routing task t to agent `ag` when `load` of the last B_WIN tasks (incl. this one) went there.
    pub fn prob(&self, t: usize, ag: usize, load: usize) -> f64 {
        let q = if self.cfg.drift && t >= self.drift_at { &self.q_post } else { &self.q_pre };
        let over = load.saturating_sub(B_CAP) as f64;
        let f = (1.0 - self.cfg.kappa * over).max(B_FLOOR);
        (q[self.ttype[t]][ag] * f).clamp(P_MIN, P_MAX)
    }

    /// reference only (ceiling diagnostics): the hidden quality table
    pub fn q_matrix(&self, post: bool) -> &Vec<Vec<f64>> {
        if post { &self.q_post } else { &self.q_pre }
    }

    /// reference only: mean of the best single-agent quality per task, ignoring congestion (an upper bound).
    pub fn oracle_nocong(&self) -> f64 {
        let mut s = 0.0;
        for t in 0..self.cfg.t {
            let q = if self.cfg.drift && t >= self.drift_at { &self.q_post } else { &self.q_pre };
            s += q[self.ttype[t]].iter().cloned().fold(0.0, f64::max);
        }
        s / self.cfg.t as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worlds_are_deterministic_and_oracle_beats_random() {
        for cfg in team_worlds() {
            let a = TeamWorld::new(3, cfg.clone());
            let b = TeamWorld::new(3, cfg.clone());
            assert_eq!(a.tasks, b.tasks);
            let team: Vec<usize> = (0..cfg.k).collect();
            assert_eq!(a.prob_at(10, &team), b.prob_at(10, &team));
            let (o, r) = a.references(3);
            if o.is_finite() {
                assert!(o > r + 0.1, "{} oracle {o} random {r}", cfg.name);
            }
            // keyword lead is a valid agent
            assert!(a.kw.values().all(|&x| x < cfg.n));
        }
    }

    #[test]
    fn congestion_lowers_success_only_when_kappa_positive() {
        let cong = AsgWorld::new(2, asg_worlds()[1].clone());
        let sep = AsgWorld::new(2, asg_worlds()[0].clone());
        assert!(cong.prob(0, 0, 12) < cong.prob(0, 0, B_CAP));
        assert_eq!(sep.prob(0, 0, 12), sep.prob(0, 0, 1));
    }
}
