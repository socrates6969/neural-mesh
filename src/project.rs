//! R46 (opt-in): PCA projector for dense vectors. Fit offline on the store's own vectors (mean + top-k eigenvectors of the covariance via
//! cyclic Jacobi, std-only), then `apply` = (v - mean) * P, L2-normalised. R14 (python, battery-gated): PCA192 + int8 = 196 B/vec (7.8x smaller than
//! f32) with no Holm loss above the .002 tolerance and real-text TOST equivalence; k <= 128 REJECTED (real_mrr -.006). Fit on tiny stores (< ~200 docs)
//! gives a full-rank rotation. Serialises with serde (json) so a fitted projector can be shipped next to the store.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projector {
    pub dim_in: usize,
    pub dim_out: usize,
    pub mean: Vec<f32>,
    /// dim_in x dim_out, row-major (row = input coordinate)
    pub p: Vec<f32>,
}

impl Projector {
    /// Fit on `n` row-major vectors of length `dim`. `k` is clamped to `dim`. Deterministic.
    pub fn fit(vectors: &[f32], dim: usize, k: usize) -> Projector {
        let n = vectors.len() / dim;
        let k = k.min(dim).max(1);
        let mut mean = vec![0f64; dim];
        for r in 0..n {
            for j in 0..dim {
                mean[j] += vectors[r * dim + j] as f64;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n.max(1) as f64);
        // covariance (dim x dim)
        let mut c = vec![0f64; dim * dim];
        let mut x = vec![0f64; dim];
        for r in 0..n {
            for j in 0..dim {
                x[j] = vectors[r * dim + j] as f64 - mean[j];
            }
            for i in 0..dim {
                let xi = x[i];
                if xi == 0.0 {
                    continue;
                }
                for j in i..dim {
                    c[i * dim + j] += xi * x[j];
                }
            }
        }
        for i in 0..dim {
            for j in i..dim {
                let v = c[i * dim + j] / n.max(1) as f64;
                c[i * dim + j] = v;
                c[j * dim + i] = v;
            }
        }
        let (vals, vecs) = jacobi_eigen(c, dim);
        let mut order: Vec<usize> = (0..dim).collect();
        order.sort_by(|&a, &b| vals[b].partial_cmp(&vals[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let mut p = vec![0f32; dim * k];
        for (col, &e) in order.iter().take(k).enumerate() {
            // fix the sign so the largest-magnitude component is positive (deterministic)
            let mut big = 0usize;
            for i in 0..dim {
                if vecs[i * dim + e].abs() > vecs[big * dim + e].abs() {
                    big = i;
                }
            }
            let sgn = if vecs[big * dim + e] < 0.0 { -1.0 } else { 1.0 };
            for i in 0..dim {
                p[i * k + col] = (sgn * vecs[i * dim + e]) as f32;
            }
        }
        Projector { dim_in: dim, dim_out: k, mean: mean.into_iter().map(|m| m as f32).collect(), p }
    }

    /// (v - mean) * P, L2-normalised (zero vector stays zero)
    pub fn apply(&self, v: &[f32]) -> Vec<f32> {
        assert_eq!(v.len(), self.dim_in, "projector input dimension");
        let k = self.dim_out;
        let mut out = vec![0f32; k];
        for i in 0..self.dim_in {
            let c = v[i] - self.mean[i];
            if c == 0.0 {
                continue;
            }
            let row = &self.p[i * k..(i + 1) * k];
            for j in 0..k {
                out[j] += c * row[j];
            }
        }
        let n = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            out.iter_mut().for_each(|x| *x /= n);
        }
        out
    }
    pub fn apply_all(&self, vectors: &[f32]) -> Vec<f32> {
        vectors.chunks_exact(self.dim_in).flat_map(|v| self.apply(v)).collect()
    }
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
    pub fn from_json(s: &str) -> Result<Projector, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// cyclic Jacobi eigenvalue algorithm for a symmetric matrix (row-major n x n). Returns (eigenvalues, eigenvectors as COLUMNS, row-major).
fn jacobi_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0f64; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..30 {
        let mut off = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    #[test]
    fn recovers_low_rank_subspace_and_preserves_neighbours() {
        // 300 vectors of dim 24 living (up to small noise) in a 3-dim subspace plus a mean offset
        let (dim, n) = (24usize, 300usize);
        let mut s = 99u64;
        let basis: Vec<Vec<f32>> = (0..3).map(|_| (0..dim).map(|_| lcg(&mut s)).collect()).collect();
        let mut data = vec![0f32; n * dim];
        for r in 0..n {
            let co: Vec<f32> = (0..3).map(|_| lcg(&mut s) * 4.0).collect();
            for j in 0..dim {
                data[r * dim + j] = 0.7 + (0..3).map(|b| co[b] * basis[b][j]).sum::<f32>() + 0.001 * lcg(&mut s);
            }
        }
        let pr = Projector::fit(&data, dim, 3);
        assert_eq!((pr.dim_in, pr.dim_out), (24, 3));
        // orthonormal columns
        for a in 0..3 {
            for b in 0..3 {
                let d: f32 = (0..dim).map(|i| pr.p[i * 3 + a] * pr.p[i * 3 + b]).sum();
                assert!((d - if a == b { 1.0 } else { 0.0 }).abs() < 1e-3, "{a},{b}: {d}");
            }
        }
        // neighbour structure preserved: nearest neighbour of vector 0 (centred cosine) is the same before/after projection
        let cen = |v: &[f32]| -> Vec<f32> {
            let mut o: Vec<f32> = v.iter().zip(&pr.mean).map(|(x, m)| x - m).collect();
            let nn = o.iter().map(|x| x * x).sum::<f32>().sqrt();
            o.iter_mut().for_each(|x| *x /= nn);
            o
        };
        let full: Vec<Vec<f32>> = (0..n).map(|r| cen(&data[r * dim..(r + 1) * dim])).collect();
        let proj: Vec<Vec<f32>> = (0..n).map(|r| pr.apply(&data[r * dim..(r + 1) * dim])).collect();
        let nn = |vs: &Vec<Vec<f32>>| {
            (1..n).max_by(|&a, &b| {
                let da: f32 = vs[0].iter().zip(&vs[a]).map(|(x, y)| x * y).sum();
                let db: f32 = vs[0].iter().zip(&vs[b]).map(|(x, y)| x * y).sum();
                da.partial_cmp(&db).unwrap()
            })
        };
        assert_eq!(nn(&full), nn(&proj));
        let rt = Projector::from_json(&pr.to_json()).unwrap();
        assert_eq!(rt, pr);
        assert!(pr.apply(&vec![0.7; dim]).iter().all(|x| x.is_finite()));
    }
}
