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


// ---------------------------------------------------------------------------
// p-values and multiple-comparison correction.

/// ln Gamma(x) for x > 0 (Lanczos, g = 7, n = 9).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction for the incomplete beta function (modified Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..400 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function I_x(a, b).
pub fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Two-sided p-value of a Student t statistic with `df` degrees of freedom.
pub fn t_two_sided_p(t: f64, df: usize) -> f64 {
    if df == 0 || t.is_nan() {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    let d = df as f64;
    inc_beta(d / 2.0, 0.5, d / (d + t * t)).clamp(0.0, 1.0)
}

/// Two-sided paired t-test p-value for H0: mean(a - b) = 0.
/// Degenerate samples: all differences identical -> p = 1 if that difference is 0, else 0.
pub fn paired_t_pvalue(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "paired_t_pvalue needs equal-length samples");
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.len();
    if n < 2 {
        return f64::NAN;
    }
    let (m, s) = (mean(&d), sd(&d));
    if s < 1e-15 {
        return if m.abs() < 1e-15 { 1.0 } else { 0.0 };
    }
    t_two_sided_p(m / (s / (n as f64).sqrt()), n - 1)
}

/// Holm-Bonferroni step-down adjusted p-values (same order as the input).
/// adj_(i) = max_{j <= i} min(1, (m - j + 1) * p_(j)) over ascending p.
/// A hypothesis is rejected at family-wise level alpha iff its adjusted p <= alpha.
pub fn holm_adjust(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut adj = vec![0.0; m];
    let mut run = 0.0f64;
    for (rank, &i) in idx.iter().enumerate() {
        let v = ((m - rank) as f64 * p[i]).min(1.0);
        run = run.max(v);
        adj[i] = run;
    }
    adj
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

    #[test]
    fn t_pvalues_match_reference_table() {
        // two-sided p at the 95% critical values must be 0.05
        for df in [1usize, 4, 9, 29, 59] {
            let p = t_two_sided_p(t_crit_95(df), df);
            assert!(close(p, 0.05, 2e-4), "df {df}: p={p}");
        }
        // t = 0 -> p = 1; huge t -> ~0
        assert!(close(t_two_sided_p(0.0, 10), 1.0, 1e-12));
        assert!(t_two_sided_p(40.0, 29) < 1e-20);
        // df = 1 is Cauchy: p = 1 - (2/pi) atan(t)
        let t = 2.0f64;
        assert!(close(t_two_sided_p(t, 1), 1.0 - 2.0 / std::f64::consts::PI * t.atan(), 1e-9));
        // symmetric in sign
        assert!(close(t_two_sided_p(-1.7, 12), t_two_sided_p(1.7, 12), 1e-12));
        // ln_gamma(5) = ln 24
        assert!(close(ln_gamma(5.0), 24f64.ln(), 1e-10));
        assert!(close(ln_gamma(0.5), std::f64::consts::PI.sqrt().ln(), 1e-10));
    }

    #[test]
    fn paired_pvalue_agrees_with_ci_decision() {
        let a = [0.6, 0.7, 0.5, 0.8];
        let b = [0.5, 0.5, 0.5, 0.6];
        let p = paired_t_pvalue(&a, &b);
        // t = 0.125 / (0.095743/2) = 2.611, df 3 -> p ~ 0.08 (CI includes 0 so p > 0.05)
        assert!(p > 0.05 && p < 0.12, "p={p}");
        let big = paired_t_pvalue(&[1.0, 1.1, 1.2, 1.3, 1.4], &[0.0; 5]);
        assert!(big < 0.01);
        assert_eq!(paired_t_pvalue(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 1.0);
        assert_eq!(paired_t_pvalue(&[2.0, 3.0, 4.0], &[1.0, 2.0, 3.0]), 0.0);
        // CI-excludes-0 <=> p < 0.05
        for k in 0..20 {
            let x: Vec<f64> = (0..10).map(|i| ((i * 7 + k) % 5) as f64 * 0.1 + 0.02 * k as f64).collect();
            let y: Vec<f64> = (0..10).map(|i| ((i * 3 + 2 * k) % 4) as f64 * 0.1).collect();
            let pd = paired_diff(&x, &y);
            let pv = paired_t_pvalue(&x, &y);
            assert_eq!(pd.significant, pv < 0.05, "k={k} p={pv} ci=[{},{}]", pd.diff.lo, pd.diff.hi);
        }
    }

    #[test]
    fn holm_hand_computed() {
        // p = [0.01, 0.04, 0.03, 0.005], m = 4
        // ascending: 0.005 (x4 = 0.02), 0.01 (x3 = 0.03), 0.03 (x2 = 0.06), 0.04 (x1 = 0.04 -> running max 0.06)
        let adj = holm_adjust(&[0.01, 0.04, 0.03, 0.005]);
        let want = [0.03, 0.06, 0.06, 0.02];
        for (a, w) in adj.iter().zip(want) {
            assert!(close(*a, w, 1e-12), "{adj:?}");
        }
    }

    #[test]
    fn holm_properties() {
        assert!(holm_adjust(&[]).is_empty());
        assert_eq!(holm_adjust(&[0.2]), vec![0.2]);
        // capped at 1, never below the raw p, monotone in raw p, never above Bonferroni
        let p = [0.9, 0.001, 0.2, 0.0004, 0.5, 0.04];
        let adj = holm_adjust(&p);
        for (a, r) in adj.iter().zip(p) {
            assert!(*a >= r && *a <= 1.0);
            assert!(*a <= (r * p.len() as f64).min(1.0) + 1e-15);
        }
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_by(|&i, &j| p[i].partial_cmp(&p[j]).unwrap());
        for w in idx.windows(2) {
            assert!(adj[w[0]] <= adj[w[1]] + 1e-15);
        }
        // [0.01, 0.02, 0.03] -> [0.03, 0.04, 0.04]
        let adj = holm_adjust(&[0.01, 0.02, 0.03]);
        assert!(close(adj[0], 0.03, 1e-12) && close(adj[1], 0.04, 1e-12) && close(adj[2], 0.04, 1e-12));
    }
}
