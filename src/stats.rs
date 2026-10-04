//! Paired statistics for the benchmark: mean, sample sd, 95% t-CI and paired
//! per-seed difference summaries. See DESIGN.md.

use serde::Serialize;

/// Two-sided 95% Student-t critical values t_{0.975, df} for df = 1..=100.
const T975: [f64; 100] = [
    12.7062, 4.3027, 3.1824, 2.7764, 2.5706, 2.4469, 2.3646, 2.3060, 2.2622, 2.2281,
    2.2010, 2.1788, 2.1604, 2.1448, 2.1314, 2.1199, 2.1098, 2.1009, 2.0930, 2.0860,
    2.0796, 2.0739, 2.0687, 2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423,
    2.0395, 2.0369, 2.0345, 2.0322, 2.0301, 2.0281, 2.0262, 2.0244, 2.0227, 2.0211,
    2.0195, 2.0181, 2.0167, 2.0154, 2.0141, 2.0129, 2.0117, 2.0106, 2.0096, 2.0086,
    2.0076, 2.0066, 2.0057, 2.0049, 2.0040, 2.0032, 2.0025, 2.0017, 2.0010, 2.0003,
    1.9996, 1.9990, 1.9983, 1.9977, 1.9971, 1.9966, 1.9960, 1.9955, 1.9949, 1.9944,
    1.9939, 1.9935, 1.9930, 1.9925, 1.9921, 1.9917, 1.9913, 1.9908, 1.9905, 1.9901,
    1.9897, 1.9893, 1.9890, 1.9886, 1.9883, 1.9879, 1.9876, 1.9873, 1.9870, 1.9867,
    1.9864, 1.9861, 1.9858, 1.9855, 1.9853, 1.9850, 1.9847, 1.9845, 1.9842, 1.9840,
];

/// t critical value for a two-sided 95% interval; falls back to 1.96 for df > 100.
pub fn t_crit_95(df: usize) -> f64 {
    if df == 0 {
        f64::NAN
    } else if df <= 100 {
        T975[df - 1]
    } else {
        1.96
    }
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (n-1 denominator). NaN for n < 2.
pub fn sd(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = mean(xs);
    (xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64).sqrt()
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    /// half-width of the 95% t-interval
    pub half_width: f64,
    pub lo: f64,
    pub hi: f64,
}

/// Mean with a 95% t confidence interval.
pub fn summarize(xs: &[f64]) -> Summary {
    let n = xs.len();
    let m = mean(xs);
    let s = sd(xs);
    let hw = if n >= 2 { t_crit_95(n - 1) * s / (n as f64).sqrt() } else { f64::NAN };
    Summary { n, mean: m, sd: s, half_width: hw, lo: m - hw, hi: m + hw }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PairedDiff {
    /// summary of a[i] - b[i]
    pub diff: Summary,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    /// true when the 95% CI excludes 0
    pub significant: bool,
}

/// Paired difference a - b over matched seeds.
pub fn paired_diff(a: &[f64], b: &[f64]) -> PairedDiff {
    assert_eq!(a.len(), b.len(), "paired_diff needs equal-length samples");
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let eps = 1e-12;
    let wins = d.iter().filter(|x| **x > eps).count();
    let losses = d.iter().filter(|x| **x < -eps).count();
    let ties = d.len() - wins - losses;
    let diff = summarize(&d);
    let significant = diff.lo > 0.0 || diff.hi < 0.0;
    PairedDiff { diff, wins, losses, ties, significant }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn mean_and_sd_hand_computed() {
        // 2,4,4,4,5,5,7,9: mean 5, sum sq dev 32, sample var 32/7
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(close(mean(&xs), 5.0, 1e-12));
        assert!(close(sd(&xs), (32.0f64 / 7.0).sqrt(), 1e-12));
        assert!(sd(&[1.0]).is_nan());
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn t_table_values() {
        assert!(close(t_crit_95(1), 12.7062, 1e-9));
        assert!(close(t_crit_95(29), 2.0452, 1e-9));
        assert!(close(t_crit_95(100), 1.9840, 1e-9));
        assert!(close(t_crit_95(500), 1.96, 1e-9));
        // monotone decreasing
        for d in 1..100 {
            assert!(t_crit_95(d) > t_crit_95(d + 1));
        }
    }

    #[test]
    fn ci_hand_computed() {
        // 1,2,3,4,5: mean 3, sd sqrt(2.5)=1.5811, se = 0.70711, t4=2.7764 -> hw 1.96322
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(s.n, 5);
        assert!(close(s.mean, 3.0, 1e-12));
        assert!(close(s.sd, 2.5f64.sqrt(), 1e-12));
        assert!(close(s.half_width, 2.7764 * 2.5f64.sqrt() / 5f64.sqrt(), 1e-12));
        assert!(close(s.half_width, 1.96322, 1e-4));
        assert!(close(s.lo, 3.0 - 1.96322, 1e-4));
    }

    #[test]
    fn paired_diff_hand_computed() {
        let a = [0.6, 0.7, 0.5, 0.8];
        let b = [0.5, 0.5, 0.5, 0.6];
        // d = 0.1, 0.2, 0.0, 0.2 : mean 0.125, var = (0.000625+0.005625+0.015625+0.005625)/3 = 0.0091667
        let p = paired_diff(&a, &b);
        assert_eq!((p.wins, p.losses, p.ties), (3, 0, 1));
        assert!(close(p.diff.mean, 0.125, 1e-12));
        assert!(close(p.diff.sd, (0.0275f64 / 3.0).sqrt(), 1e-12));
        // hw = 3.1824 * 0.095743 / 2 = 0.152347 -> CI [-0.0273, 0.2773] includes 0
        assert!(close(p.diff.half_width, 0.152347, 1e-4));
        assert!(!p.significant);

        let p2 = paired_diff(&[1.0, 1.1, 1.2, 1.3, 1.4], &[0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!(p2.significant);
        assert_eq!(p2.wins, 5);
    }
}
