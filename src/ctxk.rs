//! R90 (opt-in helper): choose the retrieval context size k PER STORE from calibration queries, instead of one global k.
//! Rule (the R77 rule, validated as a procedure in R90/R90c): the smallest k <= `cap` whose hit@k >= `frac` * hit@cap, where hit@k = share of calibration queries whose first relevant document
//! has 0-based rank < k. RD-LOG R90c (cap 5, frac .98, calibrate on one fold of a store, test on the disjoint fold): easy stores (general / domain keyword queries) pick k = 1-2 and cut context tokens
//! 60-80% at -.000..-.011 hit (not significant); hard stores (real documentation, typo'd domain) keep k = 5 EXACTLY (no change, never more tokens than the old default). Needs >= 100 calibration
//! queries (with 20 the flat curves of hard stores are mis-read: k ~ 4, -7% relative hit; with 50 -2%, with 100 <1%). A store that is hard stays at `cap`; to spend MORE on hard stores
//! (cap 8: +39..59% tokens for +.03..+.07 hit, Holm-significant) raise `cap`. This does NOT beat the fixed-k hit/token frontier on hard stores (R71/R78/R83).

/// hit@k for k = 1..=cap from the 0-based rank of the first relevant doc per calibration query (`None` = not found within the retrieved list)
pub fn hit_curve(ranks: &[Option<usize>], cap: usize) -> Vec<f64> {
    let n = ranks.len().max(1) as f64;
    (1..=cap).map(|k| ranks.iter().filter(|r| matches!(r, Some(x) if *x < k)).count() as f64 / n).collect()
}

/// smallest k in 1..=cap with hit@k >= frac * hit@cap; `None` when there are fewer than `min_queries` calibration queries, `cap == 0`, or nothing was ever found (the caller keeps its default k)
pub fn choose_k(ranks: &[Option<usize>], cap: usize, frac: f64, min_queries: usize) -> Option<usize> {
    if cap == 0 || ranks.len() < min_queries.max(1) {
        return None;
    }
    let h = hit_curve(ranks, cap);
    let top = *h.last()?;
    if top <= 0.0 {
        return None;
    }
    (1..=cap).find(|&k| h[k - 1] >= frac * top - 1e-12)
}

/// the validated defaults: cap 5, frac .98, >= 100 calibration queries
pub fn choose_k_default(ranks: &[Option<usize>]) -> Option<usize> {
    choose_k(ranks, 5, 0.98, 100)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ranks(spec: &[(Option<usize>, usize)]) -> Vec<Option<usize>> {
        spec.iter().flat_map(|(r, n)| std::iter::repeat(*r).take(*n)).collect()
    }

    #[test]
    fn easy_store_picks_small_k_hard_store_keeps_cap_and_small_samples_abstain() {
        // easy: 99 of 100 queries hit at rank 0, one at rank 1 -> hit@1 = .99 >= .98 * hit@5 (1.0) -> k = 1
        let easy = ranks(&[(Some(0), 99), (Some(1), 1)]);
        assert_eq!(choose_k_default(&easy), Some(1));
        // flat hard curve: ranks spread 0..5 + 40% misses -> every extra doc still pays -> k = cap
        let hard = ranks(&[(Some(0), 20), (Some(1), 10), (Some(2), 10), (Some(3), 10), (Some(4), 10), (None, 40)]);
        assert_eq!(choose_k_default(&hard), Some(5));
        assert_eq!(hit_curve(&hard, 5), vec![0.2, 0.3, 0.4, 0.5, 0.6]);
        // too few calibration queries, nothing found, cap 0 -> abstain (caller keeps its default)
        assert_eq!(choose_k_default(&ranks(&[(Some(0), 50)])), None);
        assert_eq!(choose_k(&ranks(&[(None, 200)]), 5, 0.98, 100), None);
        assert_eq!(choose_k(&easy, 0, 0.98, 1), None);
        // raising the cap lets a store that still gains beyond k=5 spend more
        let tail = ranks(&[(Some(0), 30), (Some(6), 30), (None, 40)]);
        assert_eq!(choose_k(&tail, 5, 0.98, 100), Some(1));
        assert_eq!(choose_k(&tail, 8, 0.98, 100), Some(7));
    }
}
