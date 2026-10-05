//! R59 (opt-in): embedding kNN capability router (calc / units / csp / sandbox / semantic / knowledge ...), the Rust twin of the python `KNNRouter` in `py/b1/lib.py`.
//! Similarity-weighted vote over the k nearest exemplars (cosine). RD-LOG: on NOVEL phrasing it reaches .88 accuracy where the hand-written keyword rules reach .36 (R41);
//! +12 diverse hand-written exemplars per class lift it to .91-.92 (R50b); k=5 is the best k (R50); BGE embeddings are much worse than MiniLM for routing (R50).
//! Exemplars are stored L2-normalised; the scan uses the multi-accumulator kernel of `flatq` (R45).

use std::collections::BTreeMap;

use crate::flatq::dot8;

#[derive(Clone, Debug)]
pub struct KnnRouter {
    dim: usize,
    k: usize,
    x: Vec<f32>,
    label: Vec<u32>,
    names: Vec<String>,
}

impl KnnRouter {
    pub fn new(dim: usize, k: usize) -> Self {
        KnnRouter { dim, k: k.max(1), x: Vec::new(), label: Vec::new(), names: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.label.len()
    }
    pub fn is_empty(&self) -> bool {
        self.label.is_empty()
    }
    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 { v.iter().map(|x| x / n).collect() } else { v.to_vec() }
    }
    /// add one labelled exemplar (a wrong-sized vector is ignored; returns whether it was added)
    pub fn add(&mut self, label: &str, v: &[f32]) -> bool {
        if v.len() != self.dim || v.iter().any(|x| !x.is_finite()) {
            return false;
        }
        let id = match self.names.iter().position(|n| n == label) {
            Some(i) => i,
            None => {
                self.names.push(label.to_string());
                self.names.len() - 1
            }
        };
        self.x.extend(Self::unit(v));
        self.label.push(id as u32);
        true
    }
    /// all labels with their vote share (sum of top-k cosines per label / total), best first. Ties keep the label whose best neighbour is nearer.
    pub fn votes(&self, q: &[f32]) -> Vec<(String, f32)> {
        self.votes_with_sim(q).0
    }
    /// like [`Self::votes`], plus the best (nearest-exemplar) cosine similarity of the query (0.0 when nothing could be scored)
    pub fn votes_with_sim(&self, q: &[f32]) -> (Vec<(String, f32)>, f32) {
        if self.is_empty() || q.len() != self.dim {
            return (vec![], 0.0);
        }
        let q = Self::unit(q);
        let mut sims: Vec<(f32, usize)> = (0..self.len()).map(|i| (dot8(&self.x[i * self.dim..(i + 1) * self.dim], &q), i)).collect();
        let k = self.k.min(sims.len());
        sims.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        sims.truncate(k);
        sims.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut v: BTreeMap<u32, (f32, usize)> = BTreeMap::new(); // label -> (vote, rank of first neighbour)
        for (rank, (s, i)) in sims.iter().enumerate() {
            let e = v.entry(self.label[*i]).or_insert((0.0, rank));
            e.0 += *s;
        }
        let tot: f32 = v.values().map(|x| x.0).sum();
        let mut out: Vec<(u32, f32, usize)> = v.into_iter().map(|(l, (s, r))| (l, s, r)).collect();
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.2.cmp(&b.2)));
        let best = sims.first().map(|x| x.0).unwrap_or(0.0);
        (out.into_iter().map(|(l, s, _)| (self.names[l as usize].clone(), if tot > 0.0 { s / tot } else { 0.0 })).collect(), best)
    }
    /// R80/R88 (opt-in safety knob): route only when the winning vote share >= `tau` AND the query is close enough to some exemplar (best cosine >= `smin`);
    /// `None` = ABSTAIN, forward to the LLM. RD-LOG: tau .6 trades +14..42% forwarded tokens for 3.5-5x fewer wrong-tool answers (R80); smin .4 changes NOTHING on the
    /// dev/blind/battery sets (in-distribution best cosine is >= .5) but removed all 9 + 7 false tool routes on 264 + 56 REAL out-of-distribution agent / human prompts (R88).
    pub fn route_guarded(&self, q: &[f32], tau: f32, smin: f32) -> Option<(String, f32)> {
        let (v, best) = self.votes_with_sim(q);
        let (l, sh) = v.into_iter().next()?;
        if sh >= tau && best >= smin { Some((l, sh)) } else { None }
    }
    /// best label and its vote share
    pub fn route(&self, q: &[f32]) -> Option<(String, f32)> {
        self.votes(q).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn votes_follow_nearest_exemplars_and_ignore_bad_input() {
        let mut r = KnnRouter::new(3, 3);
        assert!(r.route(&[1.0, 0.0, 0.0]).is_none());
        for (l, v) in [("a", [1.0, 0.1, 0.0]), ("a", [0.9, 0.0, 0.1]), ("b", [0.0, 1.0, 0.0]), ("b", [0.1, 0.9, 0.0]), ("c", [0.0, 0.0, 1.0])] {
            assert!(r.add(l, &v));
        }
        assert!(!r.add("a", &[1.0, 0.0]) && !r.add("a", &[f32::NAN, 0.0, 0.0]));
        assert_eq!(r.route(&[1.0, 0.05, 0.0]).unwrap().0, "a");
        assert_eq!(r.route(&[0.05, 1.0, 0.0]).unwrap().0, "b");
        let v = r.votes(&[0.0, 0.0, 1.0]);
        assert_eq!(v[0].0, "c".to_string().as_str());
        assert!((v.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn guarded_route_abstains_on_low_share_or_far_queries() {
        let mut r = KnnRouter::new(3, 3);
        for (l, v) in [("a", [1.0, 0.1, 0.0]), ("a", [0.9, 0.0, 0.1]), ("b", [0.0, 1.0, 0.0]), ("b", [0.1, 0.9, 0.0]), ("c", [0.0, 0.0, 1.0])] {
            r.add(l, &v);
        }
        assert_eq!(r.route_guarded(&[1.0, 0.05, 0.0], 0.6, 0.4).unwrap().0, "a");
        // a query nearly orthogonal to every exemplar (best cosine < smin) is forwarded even though some label wins the vote
        let far = [-1.0, -1.0, -1.0];
        assert!(r.route(&far).is_some() && r.route_guarded(&far, 0.0, 0.4).is_none());
        // a tie between two labels (vote share ~.5 < tau) is forwarded
        assert!(r.route_guarded(&[1.0, 1.0, 0.0], 0.8, 0.0).is_none());
        assert!(KnnRouter::new(3, 3).route_guarded(&[1.0, 0.0, 0.0], 0.0, 0.0).is_none());
    }
}
