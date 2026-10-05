//! Quantum-INSPIRED, purely classical solvers for binary assignment problems (maximise an objective).
//! Simulated annealing, parallel tempering and a "QAOA-style" layered cost/mixer schedule. They share one
//! move interface (`Prob`), so the same code anneals both problems:
//!   * `QuboK` : choose K of N agents (binary x, |x| = K), objective  sum h_i x_i + sum_{i<j} J_ij x_i x_j
//!   * `Potts` : assign each feature to one agent (one-hot binary Z), per-cell values + quadratic load penalty.
//! There is no quantum hardware or quantum speedup anywhere here; the "QAOA" arm is a classical heuristic that
//! borrows the alternating cost-layer / mixer-layer structure.
use super::world::for_each_combo;
use neural_mesh::router::Rng;

pub trait Prob {
    type S: Clone;
    fn init(&self, rng: &mut Rng) -> Self::S;
    fn propose(&self, s: &Self::S, rng: &mut Rng) -> (usize, usize);
    /// objective gain of the move (positive = better)
    fn delta(&self, s: &Self::S, m: (usize, usize)) -> f64;
    fn apply(&self, s: &mut Self::S, m: (usize, usize), d: f64);
    fn value(&self, s: &Self::S) -> f64;
}

/// Metropolis simulated annealing, geometric cooling t0 -> t1. Returns (best state, evaluations).
pub fn sa<P: Prob>(p: &P, rng: &mut Rng, iters: usize, t0: f64, t1: f64, start: Option<P::S>) -> (P::S, u64) {
    let mut s = start.unwrap_or_else(|| p.init(rng));
    let mut best = s.clone();
    let mut bv = p.value(&s);
    let ratio = (t1 / t0).powf(1.0 / iters.max(1) as f64);
    let mut t = t0;
    for _ in 0..iters {
        let m = p.propose(&s, rng);
        let d = p.delta(&s, m);
        if d >= 0.0 || rng.f64() < (d / t).exp() {
            p.apply(&mut s, m, d);
            let v = p.value(&s);
            if v > bv + 1e-12 {
                bv = v;
                best = s.clone();
            }
        }
        t *= ratio;
    }
    (best, iters as u64)
}

/// Parallel tempering: `reps` replicas on a geometric ladder t_lo..t_hi, 3 Metropolis moves per replica per round,
/// then neighbour exchanges. Total move evaluations ~ iters.
pub fn pt<P: Prob>(p: &P, rng: &mut Rng, iters: usize, reps: usize, t_lo: f64, t_hi: f64, start: Option<P::S>) -> (P::S, u64) {
    let reps = reps.max(2);
    let temps: Vec<f64> = (0..reps).map(|r| t_lo * (t_hi / t_lo).powf(r as f64 / (reps - 1) as f64)).collect();
    let mut st: Vec<P::S> = (0..reps).map(|r| if r == 0 && start.is_some() { start.clone().unwrap() } else { p.init(rng) }).collect();
    let mut best = st[0].clone();
    let mut bv = p.value(&best);
    for s in &st {
        let v = p.value(s);
        if v > bv {
            bv = v;
            best = s.clone();
        }
    }
    let m = 3usize;
    let rounds = (iters / (reps * m)).max(1);
    let mut evals = 0u64;
    for _ in 0..rounds {
        for r in 0..reps {
            for _ in 0..m {
                let mv = p.propose(&st[r], rng);
                let d = p.delta(&st[r], mv);
                evals += 1;
                if d >= 0.0 || rng.f64() < (d / temps[r]).exp() {
                    p.apply(&mut st[r], mv, d);
                    let v = p.value(&st[r]);
                    if v > bv + 1e-12 {
                        bv = v;
                        best = st[r].clone();
                    }
                }
            }
        }
        for r in 0..reps - 1 {
            let (va, vb) = (p.value(&st[r]), p.value(&st[r + 1]));
            let a = (vb - va) * (1.0 / temps[r] - 1.0 / temps[r + 1]);
            if a >= 0.0 || rng.f64() < a.exp() {
                st.swap(r, r + 1);
            }
        }
    }
    (best, evals)
}

/// QAOA-STYLE classical heuristic. `shots` independent states go through `layers` alternating layers:
/// a cost layer (heat-bath accepted moves with inverse temperature gamma_l, linear ramp g_lo..g_hi, the
/// "linear-ramp" QAOA schedule) and a mixer layer (a forced random move with probability sin^2(beta_l),
/// beta_l ramping down). Best state ever seen wins. This is annealing with a population and a
/// mixer; it carries no quantum advantage.
pub fn qaoa<P: Prob>(p: &P, rng: &mut Rng, iters: usize, shots: usize, layers: usize, g_lo: f64, g_hi: f64, beta_max: f64, start: Option<P::S>) -> (P::S, u64) {
    let shots = shots.max(1);
    let layers = layers.max(1);
    let m = (iters / (shots * layers)).max(1);
    let mut st: Vec<P::S> = (0..shots).map(|i| if i == 0 && start.is_some() { start.clone().unwrap() } else { p.init(rng) }).collect();
    let mut best = st[0].clone();
    let mut bv = p.value(&best);
    let mut evals = 0u64;
    for l in 0..layers {
        let frac = (l as f64 + 0.5) / layers as f64;
        let gamma = g_lo + (g_hi - g_lo) * frac;
        let beta = beta_max * (1.0 - frac);
        let pmix = beta.sin().powi(2);
        for s in st.iter_mut() {
            for _ in 0..m {
                let mv = p.propose(s, rng);
                let d = p.delta(s, mv);
                evals += 1;
                if rng.f64() < 1.0 / (1.0 + (-gamma * d).exp()) {
                    p.apply(s, mv, d);
                    let v = p.value(s);
                    if v > bv + 1e-12 {
                        bv = v;
                        best = s.clone();
                    }
                }
                if rng.f64() < pmix {
                    let mv = p.propose(s, rng);
                    let d = p.delta(s, mv);
                    evals += 1;
                    p.apply(s, mv, d);
                }
            }
        }
    }
    (best, evals)
}

// ------------------------------------------------------------------ QUBO with cardinality K
pub struct QuboK {
    pub n: usize,
    pub k: usize,
    pub h: Vec<f64>,
    /// symmetric n*n, zero diagonal
    pub j: Vec<f64>,
}

#[derive(Clone)]
pub struct QS {
    pub members: Vec<usize>,
    inx: Vec<bool>,
    phi: Vec<f64>,
    v: f64,
}

impl QuboK {
    pub fn eval(&self, team: &[usize]) -> f64 {
        let mut v = 0.0;
        for (x, &a) in team.iter().enumerate() {
            v += self.h[a];
            for &b in &team[x + 1..] {
                v += self.j[a * self.n + b];
            }
        }
        v
    }
    pub fn state_from(&self, team: &[usize]) -> QS {
        let mut inx = vec![false; self.n];
        for &a in team {
            inx[a] = true;
        }
        let phi: Vec<f64> = (0..self.n).map(|i| self.h[i] + team.iter().map(|&k| self.j[i * self.n + k]).sum::<f64>()).collect();
        QS { members: team.to_vec(), inx, phi, v: self.eval(team) }
    }
    /// exhaustive search (use only when C(n,k) is small)
    pub fn exact(&self) -> (Vec<usize>, u64) {
        let mut best = (f64::NEG_INFINITY, vec![]);
        let mut evals = 0u64;
        for_each_combo(self.n, self.k, |c| {
            evals += 1;
            let v = self.eval(c);
            if v > best.0 {
                best = (v, c.to_vec());
            }
        });
        (best.1, evals)
    }
    /// forward greedy by marginal gain, then steepest-ascent swap polish
    pub fn greedy(&self) -> (Vec<usize>, u64) {
        let mut team: Vec<usize> = vec![];
        let mut evals = 0u64;
        for _ in 0..self.k {
            let mut best = (f64::NEG_INFINITY, 0usize);
            for b in 0..self.n {
                if team.contains(&b) {
                    continue;
                }
                evals += 1;
                let g = self.h[b] + team.iter().map(|&a| self.j[a * self.n + b]).sum::<f64>();
                if g > best.0 {
                    best = (g, b);
                }
            }
            team.push(best.1);
        }
        let mut s = self.state_from(&team);
        loop {
            let mut bm = (1e-12, None);
            for pos in 0..self.k {
                for b in 0..self.n {
                    if s.inx[b] {
                        continue;
                    }
                    evals += 1;
                    let d = self.delta(&s, (pos, b));
                    if d > bm.0 {
                        bm = (d, Some((pos, b)));
                    }
                }
            }
            match bm.1 {
                Some(m) => self.apply(&mut s, m, bm.0),
                None => break,
            }
        }
        (s.members, evals)
    }
}

impl Prob for QuboK {
    type S = QS;
    fn init(&self, rng: &mut Rng) -> QS {
        let mut ag: Vec<usize> = (0..self.n).collect();
        for i in (1..ag.len()).rev() {
            let j = rng.below(i + 1);
            ag.swap(i, j);
        }
        self.state_from(&ag[..self.k])
    }
    fn propose(&self, s: &QS, rng: &mut Rng) -> (usize, usize) {
        let pos = rng.below(self.k);
        loop {
            let b = rng.below(self.n);
            if !s.inx[b] {
                return (pos, b);
            }
        }
    }
    fn delta(&self, s: &QS, m: (usize, usize)) -> f64 {
        let a = s.members[m.0];
        let b = m.1;
        (s.phi[b] - self.j[a * self.n + b]) - s.phi[a]
    }
    fn apply(&self, s: &mut QS, m: (usize, usize), d: f64) {
        let a = s.members[m.0];
        let b = m.1;
        for i in 0..self.n {
            s.phi[i] += self.j[i * self.n + b] - self.j[i * self.n + a];
        }
        s.inx[a] = false;
        s.inx[b] = true;
        s.members[m.0] = b;
        s.v += d;
    }
    fn value(&self, s: &QS) -> f64 {
        s.v
    }
}

// ------------------------------------------------------------------ Potts / one-hot assignment with load penalty
pub struct Potts {
    pub nf: usize,
    pub na: usize,
    /// nf*na cell values (already traffic-weighted)
    pub val: Vec<f64>,
    /// traffic share per feature (sums to 1)
    pub w: Vec<f64>,
    pub cap: f64,
    pub lam: f64,
}

#[derive(Clone)]
pub struct PS {
    pub assign: Vec<usize>,
    load: Vec<f64>,
    v: f64,
}

impl Potts {
    fn pen(&self, l: f64) -> f64 {
        let o = (l - self.cap).max(0.0);
        self.lam * o * o
    }
    pub fn state_from(&self, assign: &[usize]) -> PS {
        let mut load = vec![0.0; self.na];
        let mut v = 0.0;
        for f in 0..self.nf {
            load[assign[f]] += self.w[f];
            v += self.val[f * self.na + assign[f]];
        }
        for a in 0..self.na {
            v -= self.pen(load[a]);
        }
        PS { assign: assign.to_vec(), load, v }
    }
    /// decoupled argmax (the exact optimum when lam = 0)
    pub fn argmax(&self) -> PS {
        let asg: Vec<usize> = (0..self.nf)
            .map(|f| {
                (0..self.na).max_by(|&a, &b| self.val[f * self.na + a].total_cmp(&self.val[f * self.na + b])).unwrap()
            })
            .collect();
        self.state_from(&asg)
    }
    /// steepest single-feature reassignment polish
    pub fn polish(&self, mut s: PS) -> (PS, u64) {
        let mut evals = 0u64;
        loop {
            let mut bm = (1e-12, None);
            for f in 0..self.nf {
                for b in 0..self.na {
                    if b == s.assign[f] {
                        continue;
                    }
                    evals += 1;
                    let d = self.delta(&s, (f, b));
                    if d > bm.0 {
                        bm = (d, Some((f, b)));
                    }
                }
            }
            match bm.1 {
                Some(m) => self.apply(&mut s, m, bm.0),
                None => break,
            }
        }
        (s, evals)
    }
}

impl Prob for Potts {
    type S = PS;
    fn init(&self, rng: &mut Rng) -> PS {
        let asg: Vec<usize> = (0..self.nf).map(|_| rng.below(self.na)).collect();
        self.state_from(&asg)
    }
    fn propose(&self, s: &PS, rng: &mut Rng) -> (usize, usize) {
        let f = rng.below(self.nf);
        let mut b = rng.below(self.na - 1);
        if b >= s.assign[f] {
            b += 1;
        }
        (f, b)
    }
    fn delta(&self, s: &PS, m: (usize, usize)) -> f64 {
        let (f, b) = m;
        let a = s.assign[f];
        let w = self.w[f];
        self.val[f * self.na + b] - self.val[f * self.na + a]
            - (self.pen(s.load[b] + w) - self.pen(s.load[b]) + self.pen(s.load[a] - w) - self.pen(s.load[a]))
    }
    fn apply(&self, s: &mut PS, m: (usize, usize), d: f64) {
        let (f, b) = m;
        let a = s.assign[f];
        s.load[a] -= self.w[f];
        s.load[b] += self.w[f];
        s.assign[f] = b;
        s.v += d;
    }
    fn value(&self, s: &PS) -> f64 {
        s.v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_qubo(n: usize, k: usize, seed: u64) -> QuboK {
        let mut r = Rng::new(seed);
        let h: Vec<f64> = (0..n).map(|_| r.f64() * 0.2).collect();
        let mut j = vec![0.0; n * n];
        for a in 0..n {
            for b in a + 1..n {
                let v = (r.f64() - 0.5) * 0.4;
                j[a * n + b] = v;
                j[b * n + a] = v;
            }
        }
        QuboK { n, k, h, j }
    }

    #[test]
    fn solvers_reach_exact_optimum_most_of_the_time() {
        let (mut sa_ok, mut pt_ok, mut qa_ok, mut gr_ok) = (0, 0, 0, 0);
        let trials = 40;
        for seed in 0..trials {
            let q = rand_qubo(12, 4, seed as u64 + 1);
            let (ex, _) = q.exact();
            let ev = q.eval(&ex);
            let mut r = Rng::new(seed as u64 + 77);
            let (s, _) = sa(&q, &mut r, 400, 0.1, 0.005, None);
            if q.eval(&s.members) >= ev - 1e-9 {
                sa_ok += 1;
            }
            let (s, _) = pt(&q, &mut r, 600, 6, 0.01, 0.15, None);
            if q.eval(&s.members) >= ev - 1e-9 {
                pt_ok += 1;
            }
            let (s, _) = qaoa(&q, &mut r, 600, 6, 6, 8.0, 150.0, 0.4, None);
            if q.eval(&s.members) >= ev - 1e-9 {
                qa_ok += 1;
            }
            let (g, _) = q.greedy();
            if q.eval(&g) >= ev - 1e-9 {
                gr_ok += 1;
            }
            // incremental state consistent with full eval
            assert!((q.state_from(&s.members).v - q.eval(&s.members)).abs() < 1e-9);
        }
        assert!(sa_ok >= 30 && pt_ok >= 35 && qa_ok >= 35 && gr_ok >= 20, "sa {sa_ok} pt {pt_ok} qaoa {qa_ok} greedy {gr_ok}");
    }

    #[test]
    fn incremental_delta_matches_full_eval() {
        let q = rand_qubo(10, 4, 5);
        let mut r = Rng::new(3);
        let mut s = q.init(&mut r);
        for _ in 0..200 {
            let m = q.propose(&s, &mut r);
            let d = q.delta(&s, m);
            let before = q.eval(&s.members);
            q.apply(&mut s, m, d);
            assert!((q.eval(&s.members) - before - d).abs() < 1e-9);
        }
    }

    #[test]
    fn potts_delta_and_anneal_beat_argmax_under_load() {
        let (nf, na) = (12, 4);
        let mut r = Rng::new(9);
        let val: Vec<f64> = (0..nf * na).map(|i| if i % na == 0 { 0.1 } else { 0.08 * r.f64() }).collect();
        let w = vec![1.0 / nf as f64; nf];
        let p = Potts { nf, na, val, w, cap: 0.4, lam: 2.0 };
        let am = p.argmax();
        let mut rr = Rng::new(1);
        let mut s = p.init(&mut rr);
        for _ in 0..300 {
            let m = p.propose(&s, &mut rr);
            let d = p.delta(&s, m);
            let before = p.state_from(&s.assign).v;
            p.apply(&mut s, m, d);
            assert!((p.state_from(&s.assign).v - before - d).abs() < 1e-9);
        }
        let (best, _) = sa(&p, &mut rr, 3000, 0.05, 0.001, Some(am.clone()));
        assert!(best.v >= am.v - 1e-12);
        assert!(best.v > am.v + 1e-6, "annealing should improve on the congested argmax");
    }

    #[test]
    fn combos_count() {
        let mut c = 0;
        for_each_combo(8, 3, |_| c += 1);
        assert_eq!(c, 56);
        let mut c = 0;
        for_each_combo(5, 5, |_| c += 1);
        assert_eq!(c, 1);
    }
}
