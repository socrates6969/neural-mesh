//! Spatial memory: 2D/3D point indexes (k-d tree, uniform grid), AABB layout
//! overlap, grid path finding (A*), and a few "visual-spatial" task features.
//!
//! Everything is `std`-only, deterministic, and generic over the dimension `D`
//! (use `D = 2` or `D = 3`). Every query has a brute-force twin used by the
//! property tests and the benchmark (`src/bin/r3_spatial.rs`).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

pub type Pt<const D: usize> = [f64; D];

#[inline]
pub fn dist2<const D: usize>(a: &Pt<D>, b: &Pt<D>) -> f64 {
    let mut s = 0.0;
    for i in 0..D {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[inline]
pub fn manhattan<const D: usize>(a: &Pt<D>, b: &Pt<D>) -> f64 {
    (0..D).map(|i| (a[i] - b[i]).abs()).sum()
}

/// `f64` with a total order (NaN sorts last) for heaps.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Of(f64);
impl Eq for Of {}
impl PartialOrd for Of {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Of {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Brute force reference
// ---------------------------------------------------------------------------

/// `(index, squared distance)` of the nearest point; ties -> lowest index.
pub fn brute_nearest<const D: usize>(pts: &[Pt<D>], q: &Pt<D>) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, p) in pts.iter().enumerate() {
        let d = dist2(p, q);
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best
}

/// k nearest `(index, squared distance)`, ascending distance (ties by index).
pub fn brute_knn<const D: usize>(pts: &[Pt<D>], q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = pts.iter().enumerate().map(|(i, p)| (i, dist2(p, q))).collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

/// Indices of points inside the closed box `[lo, hi]`, sorted.
pub fn brute_range<const D: usize>(pts: &[Pt<D>], lo: &Pt<D>, hi: &Pt<D>) -> Vec<usize> {
    pts.iter()
        .enumerate()
        .filter(|(_, p)| (0..D).all(|i| p[i] >= lo[i] && p[i] <= hi[i]))
        .map(|(i, _)| i)
        .collect()
}

/// Indices of points within Euclidean `radius` (closed), sorted.
pub fn brute_radius<const D: usize>(pts: &[Pt<D>], q: &Pt<D>, radius: f64) -> Vec<usize> {
    let r2 = radius * radius;
    pts.iter().enumerate().filter(|(_, p)| dist2(p, q) <= r2).map(|(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// k-d tree (implicit, balanced: the median of each slice is the node)
// ---------------------------------------------------------------------------

pub struct KdTree<const D: usize> {
    pts: Vec<Pt<D>>,
    /// permutation of point indices laid out as an implicit balanced tree
    idx: Vec<u32>,
}

impl<const D: usize> KdTree<D> {
    pub fn build(pts: &[Pt<D>]) -> Self {
        let mut idx: Vec<u32> = (0..pts.len() as u32).collect();
        let pv = pts.to_vec();
        Self::build_rec(&pv, &mut idx[..], 0);
        KdTree { pts: pv, idx }
    }
    fn build_rec(pts: &[Pt<D>], idx: &mut [u32], depth: usize) {
        if idx.len() <= 1 {
            return;
        }
        let ax = depth % D;
        let mid = idx.len() / 2;
        idx.select_nth_unstable_by(mid, |&a, &b| {
            pts[a as usize][ax].partial_cmp(&pts[b as usize][ax]).unwrap_or(Ordering::Equal)
        });
        let (l, r) = idx.split_at_mut(mid);
        Self::build_rec(pts, l, depth + 1);
        Self::build_rec(pts, &mut r[1..], depth + 1);
    }
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }
    /// Approximate resident bytes.
    pub fn bytes(&self) -> usize {
        self.pts.len() * D * 8 + self.idx.len() * 4
    }

    pub fn nearest(&self, q: &Pt<D>) -> Option<(usize, f64)> {
        let r = self.knn(q, 1);
        r.into_iter().next()
    }

    /// k nearest `(index, squared distance)`, ascending.
    pub fn knn(&self, q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.pts.is_empty() {
            return vec![];
        }
        let mut heap: BinaryHeap<(Of, u32)> = BinaryHeap::with_capacity(k + 1);
        self.knn_rec(&self.idx[..], 0, q, k, &mut heap);
        let mut v: Vec<(usize, f64)> = heap.into_iter().map(|(d, i)| (i as usize, d.0)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
        v
    }
    fn knn_rec(&self, idx: &[u32], depth: usize, q: &Pt<D>, k: usize, heap: &mut BinaryHeap<(Of, u32)>) {
        if idx.is_empty() {
            return;
        }
        let ax = depth % D;
        let mid = idx.len() / 2;
        let pi = idx[mid];
        let p = &self.pts[pi as usize];
        let d = dist2(p, q);
        if heap.len() < k {
            heap.push((Of(d), pi));
        } else if let Some(&(Of(worst), wi)) = heap.peek() {
            // tie-break toward lower index so results are deterministic
            if d < worst || (d == worst && pi < wi) {
                heap.pop();
                heap.push((Of(d), pi));
            }
        }
        let diff = q[ax] - p[ax];
        let (near, far) = if diff < 0.0 { (&idx[..mid], &idx[mid + 1..]) } else { (&idx[mid + 1..], &idx[..mid]) };
        self.knn_rec(near, depth + 1, q, k, heap);
        let worst = heap.peek().map(|x| x.0 .0).unwrap_or(f64::INFINITY);
        if heap.len() < k || diff * diff <= worst {
            self.knn_rec(far, depth + 1, q, k, heap);
        }
    }

    /// Points in the closed box `[lo, hi]`, sorted by index.
    pub fn range(&self, lo: &Pt<D>, hi: &Pt<D>) -> Vec<usize> {
        let mut out = vec![];
        self.range_rec(&self.idx[..], 0, lo, hi, &mut out);
        out.sort_unstable();
        out
    }
    fn range_rec(&self, idx: &[u32], depth: usize, lo: &Pt<D>, hi: &Pt<D>, out: &mut Vec<usize>) {
        if idx.is_empty() {
            return;
        }
        let ax = depth % D;
        let mid = idx.len() / 2;
        let pi = idx[mid] as usize;
        let p = &self.pts[pi];
        if (0..D).all(|i| p[i] >= lo[i] && p[i] <= hi[i]) {
            out.push(pi);
        }
        if lo[ax] <= p[ax] {
            self.range_rec(&idx[..mid], depth + 1, lo, hi, out);
        }
        if hi[ax] >= p[ax] {
            self.range_rec(&idx[mid + 1..], depth + 1, lo, hi, out);
        }
    }

    /// Points within Euclidean `radius`, sorted by index.
    pub fn radius(&self, q: &Pt<D>, radius: f64) -> Vec<usize> {
        let mut lo = *q;
        let mut hi = *q;
        for i in 0..D {
            lo[i] -= radius;
            hi[i] += radius;
        }
        let r2 = radius * radius;
        self.range(&lo, &hi).into_iter().filter(|&i| dist2(&self.pts[i], q) <= r2).collect()
    }
}

// ---------------------------------------------------------------------------
// Uniform grid
// ---------------------------------------------------------------------------

pub struct Grid<const D: usize> {
    pts: Vec<Pt<D>>,
    origin: Pt<D>,
    cell: f64,
    dims: [usize; D],
    cells: Vec<Vec<u32>>,
}

impl<const D: usize> Grid<D> {
    /// Build over the bounding box of `pts` with cell edge `cell` (> 0).
    pub fn build(pts: &[Pt<D>], cell: f64) -> Self {
        assert!(cell > 0.0);
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for p in pts {
            for i in 0..D {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        if pts.is_empty() {
            lo = [0.0; D];
            hi = [0.0; D];
        }
        let mut dims = [1usize; D];
        for i in 0..D {
            dims[i] = (((hi[i] - lo[i]) / cell).floor() as usize + 1).max(1);
        }
        let total: usize = dims.iter().product();
        let mut g = Grid { pts: pts.to_vec(), origin: lo, cell, dims, cells: vec![Vec::new(); total] };
        for (i, p) in pts.iter().enumerate() {
            let c = g.cell_of(p);
            let f = g.flat(&c);
            g.cells[f].push(i as u32);
        }
        g
    }
    /// A cell size giving about `per_cell` points per cell for roughly uniform data.
    pub fn auto_cell(pts: &[Pt<D>], per_cell: f64) -> f64 {
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for p in pts {
            for i in 0..D {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        let vol: f64 = (0..D).map(|i| (hi[i] - lo[i]).max(1e-9)).product();
        let n = pts.len().max(1) as f64;
        (vol * per_cell / n).powf(1.0 / D as f64).max(1e-9)
    }
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.pts.len() * (D * 8 + 4) + self.cells.len() * 24
    }
    fn cell_of(&self, p: &Pt<D>) -> [usize; D] {
        let mut c = [0usize; D];
        for i in 0..D {
            let f = ((p[i] - self.origin[i]) / self.cell).floor();
            c[i] = if f < 0.0 { 0 } else { (f as usize).min(self.dims[i] - 1) };
        }
        c
    }
    fn flat(&self, c: &[usize; D]) -> usize {
        let mut f = 0;
        for i in 0..D {
            f = f * self.dims[i] + c[i];
        }
        f
    }

    /// Visit every cell with Chebyshev distance exactly `r` from `c`.
    fn for_ring(&self, c: &[usize; D], r: usize, f: &mut dyn FnMut(usize)) {
        let mut cur = [0i64; D];
        self.ring_rec(c, r as i64, 0, &mut cur, false, f);
    }
    fn ring_rec(&self, c: &[usize; D], r: i64, ax: usize, cur: &mut [i64; D], on_edge: bool, f: &mut dyn FnMut(usize)) {
        if ax == D {
            if on_edge || r == 0 {
                let mut u = [0usize; D];
                for i in 0..D {
                    u[i] = cur[i] as usize;
                }
                f(self.flat(&u));
            }
            return;
        }
        let lo = (c[ax] as i64 - r).max(0);
        let hi = (c[ax] as i64 + r).min(self.dims[ax] as i64 - 1);
        for v in lo..=hi {
            cur[ax] = v;
            let edge = on_edge || (v - c[ax] as i64).abs() == r;
            self.ring_rec(c, r, ax + 1, cur, edge, f);
        }
    }

    pub fn knn(&self, q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.pts.is_empty() {
            return vec![];
        }
        let c = self.cell_of(q);
        let maxr = *self.dims.iter().max().unwrap();
        let mut heap: BinaryHeap<(Of, u32)> = BinaryHeap::with_capacity(k + 1);
        for r in 0..=maxr {
            let pts = &self.pts;
            let cells = &self.cells;
            self.for_ring(&c, r, &mut |fi| {
                for &pi in &cells[fi] {
                    let d = dist2(&pts[pi as usize], q);
                    if heap.len() < k {
                        heap.push((Of(d), pi));
                    } else if let Some(&(Of(w), wi)) = heap.peek() {
                        if d < w || (d == w && pi < wi) {
                            heap.pop();
                            heap.push((Of(d), pi));
                        }
                    }
                }
            });
            if heap.len() == k {
                let w = heap.peek().unwrap().0 .0;
                let lim = r as f64 * self.cell;
                if w <= lim * lim {
                    break;
                }
            }
        }
        let mut v: Vec<(usize, f64)> = heap.into_iter().map(|(d, i)| (i as usize, d.0)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
        v
    }

    pub fn nearest(&self, q: &Pt<D>) -> Option<(usize, f64)> {
        self.knn(q, 1).into_iter().next()
    }

    pub fn range(&self, lo: &Pt<D>, hi: &Pt<D>) -> Vec<usize> {
        let a = self.cell_of(lo);
        let b = self.cell_of(hi);
        let mut out = vec![];
        let mut cur = a;
        loop {
            for &pi in &self.cells[self.flat(&cur)] {
                let p = &self.pts[pi as usize];
                if (0..D).all(|i| p[i] >= lo[i] && p[i] <= hi[i]) {
                    out.push(pi as usize);
                }
            }
            // odometer increment
            let mut ax = D;
            loop {
                if ax == 0 {
                    out.sort_unstable();
                    return out;
                }
                ax -= 1;
                if cur[ax] < b[ax] {
                    cur[ax] += 1;
                    break;
                }
                cur[ax] = a[ax];
            }
        }
    }

    pub fn radius(&self, q: &Pt<D>, radius: f64) -> Vec<usize> {
        let mut lo = *q;
        let mut hi = *q;
        for i in 0..D {
            lo[i] -= radius;
            hi[i] += radius;
        }
        let r2 = radius * radius;
        self.range(&lo, &hi).into_iter().filter(|&i| dist2(&self.pts[i], q) <= r2).collect()
    }
}

// ---------------------------------------------------------------------------
// AABB layout overlap
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb<const D: usize> {
    pub lo: Pt<D>,
    pub hi: Pt<D>,
}

impl<const D: usize> Aabb<D> {
    pub fn new(lo: Pt<D>, hi: Pt<D>) -> Self {
        Aabb { lo, hi }
    }
    /// Closed-box intersection (touching boxes intersect).
    pub fn intersects(&self, o: &Aabb<D>) -> bool {
        (0..D).all(|i| self.lo[i] <= o.hi[i] && o.lo[i] <= self.hi[i])
    }
    pub fn volume(&self) -> f64 {
        (0..D).map(|i| (self.hi[i] - self.lo[i]).max(0.0)).product()
    }
    pub fn intersection_volume(&self, o: &Aabb<D>) -> f64 {
        (0..D).map(|i| (self.hi[i].min(o.hi[i]) - self.lo[i].max(o.lo[i])).max(0.0)).product()
    }
}

/// All intersecting pairs `(i < j)`, sorted. O(n^2).
pub fn overlap_pairs_brute<const D: usize>(b: &[Aabb<D>]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for i in 0..b.len() {
        for j in i + 1..b.len() {
            if b[i].intersects(&b[j]) {
                out.push((i, j));
            }
        }
    }
    out
}

/// Sweep-and-prune on axis 0: same result as the brute force, sorted.
pub fn overlap_pairs_sweep<const D: usize>(b: &[Aabb<D>]) -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = (0..b.len()).collect();
    order.sort_by(|&x, &y| b[x].lo[0].partial_cmp(&b[y].lo[0]).unwrap_or(Ordering::Equal));
    let mut out = vec![];
    for (pos, &i) in order.iter().enumerate() {
        for &j in &order[pos + 1..] {
            if b[j].lo[0] > b[i].hi[0] {
                break;
            }
            if b[i].intersects(&b[j]) {
                out.push((i.min(j), i.max(j)));
            }
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// Grid path finding
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GridMap {
    pub w: usize,
    pub h: usize,
    pub blocked: Vec<bool>,
}

impl GridMap {
    pub fn new(w: usize, h: usize) -> Self {
        GridMap { w, h, blocked: vec![false; w * h] }
    }
    /// Random obstacles with the given density, never blocking `keep_free` cells.
    pub fn random(w: usize, h: usize, density: f64, rng: &mut crate::router::Rng, keep_free: &[(usize, usize)]) -> Self {
        let mut m = GridMap::new(w, h);
        for i in 0..w * h {
            m.blocked[i] = rng.bernoulli(density);
        }
        for &(x, y) in keep_free {
            m.blocked[y * w + x] = false;
        }
        m
    }
    #[inline]
    fn open(&self, x: usize, y: usize) -> bool {
        !self.blocked[y * self.w + x]
    }
    fn neighbours(&self, x: usize, y: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut v = [(usize::MAX, usize::MAX); 4];
        let mut n = 0;
        if x > 0 {
            v[n] = (x - 1, y);
            n += 1;
        }
        if x + 1 < self.w {
            v[n] = (x + 1, y);
            n += 1;
        }
        if y > 0 {
            v[n] = (x, y - 1);
            n += 1;
        }
        if y + 1 < self.h {
            v[n] = (x, y + 1);
            n += 1;
        }
        v.into_iter().take(n).filter(move |&(a, b)| self.open(a, b))
    }

    /// Shortest 4-connected path length (steps) by BFS; the reference.
    pub fn bfs_len(&self, s: (usize, usize), g: (usize, usize)) -> Option<usize> {
        if !self.open(s.0, s.1) || !self.open(g.0, g.1) {
            return None;
        }
        let mut dist = vec![usize::MAX; self.w * self.h];
        let mut dq = VecDeque::new();
        dist[s.1 * self.w + s.0] = 0;
        dq.push_back(s);
        while let Some((x, y)) = dq.pop_front() {
            let d = dist[y * self.w + x];
            if (x, y) == g {
                return Some(d);
            }
            for (a, b) in self.neighbours(x, y) {
                if dist[b * self.w + a] == usize::MAX {
                    dist[b * self.w + a] = d + 1;
                    dq.push_back((a, b));
                }
            }
        }
        None
    }

    /// A* with the Manhattan heuristic (admissible and consistent on a
    /// 4-connected unit-cost grid). Returns `(length, nodes_expanded)`.
    pub fn astar_len(&self, s: (usize, usize), g: (usize, usize)) -> Option<(usize, usize)> {
        if !self.open(s.0, s.1) || !self.open(g.0, g.1) {
            return None;
        }
        let hfn = |x: usize, y: usize| x.abs_diff(g.0) + y.abs_diff(g.1);
        let mut best = vec![usize::MAX; self.w * self.h];
        let mut heap: BinaryHeap<std::cmp::Reverse<(usize, usize, usize, usize)>> = BinaryHeap::new();
        best[s.1 * self.w + s.0] = 0;
        heap.push(std::cmp::Reverse((hfn(s.0, s.1), 0, s.0, s.1)));
        let mut expanded = 0;
        while let Some(std::cmp::Reverse((_f, gc, x, y))) = heap.pop() {
            if gc > best[y * self.w + x] {
                continue;
            }
            expanded += 1;
            if (x, y) == g {
                return Some((gc, expanded));
            }
            for (a, b) in self.neighbours(x, y) {
                let ng = gc + 1;
                if ng < best[b * self.w + a] {
                    best[b * self.w + a] = ng;
                    heap.push(std::cmp::Reverse((ng + hfn(a, b), ng, a, b)));
                }
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Visual-spatial task features
// ---------------------------------------------------------------------------

/// Discretised, deterministic features of a 2D layout (what a router could see
/// if tasks carried a layout): overlap-density bucket and box-count bucket.
pub fn layout_features(boxes: &[Aabb<2>]) -> Vec<String> {
    let n = boxes.len();
    let pairs = overlap_pairs_sweep(boxes).len();
    let dens = if n < 2 { 0.0 } else { pairs as f64 / n as f64 };
    let d = if dens < 0.5 {
        "low"
    } else if dens < 1.5 {
        "mid"
    } else {
        "high"
    };
    let c = if n < 20 { "few" } else if n < 60 { "some" } else { "many" };
    vec![format!("sp:overlap-{d}"), format!("sp:count-{c}")]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::Rng;

    fn rand_pts<const D: usize>(rng: &mut Rng, n: usize, clustered: bool) -> Vec<Pt<D>> {
        (0..n)
            .map(|_| {
                let mut p = [0.0; D];
                for i in 0..D {
                    p[i] = if clustered { (rng.f64() * 8.0).floor() + rng.f64() * 0.01 } else { rng.f64() * 100.0 };
                }
                p
            })
            .collect()
    }

    fn check_knn<const D: usize>(seed: u64, clustered: bool) {
        let mut rng = Rng::new(seed);
        let pts: Vec<Pt<D>> = rand_pts(&mut rng, 300, clustered);
        let kd = KdTree::build(&pts);
        let cell = Grid::<D>::auto_cell(&pts, 3.0);
        let gr = Grid::build(&pts, cell);
        for _ in 0..60 {
            let mut q = [0.0; D];
            for i in 0..D {
                q[i] = rng.f64() * 120.0 - 10.0;
            }
            let k = 1 + rng.below(12);
            let b = brute_knn(&pts, &q, k);
            for got in [kd.knn(&q, k), gr.knn(&q, k)] {
                assert_eq!(got.len(), b.len());
                for (x, y) in got.iter().zip(&b) {
                    assert!((x.1 - y.1).abs() < 1e-9, "knn dist mismatch {x:?} {y:?}");
                }
            }
            let bn = brute_nearest(&pts, &q).unwrap();
            assert!((kd.nearest(&q).unwrap().1 - bn.1).abs() < 1e-9);
            assert!((gr.nearest(&q).unwrap().1 - bn.1).abs() < 1e-9);
        }
    }

    #[test]
    fn knn_matches_brute_2d_3d() {
        for s in 1..6 {
            check_knn::<2>(s, false);
            check_knn::<3>(s, false);
            check_knn::<2>(s + 100, true);
            check_knn::<3>(s + 100, true);
        }
    }

    fn check_range<const D: usize>(seed: u64) {
        let mut rng = Rng::new(seed);
        let pts: Vec<Pt<D>> = rand_pts(&mut rng, 400, false);
        let kd = KdTree::build(&pts);
        let gr = Grid::build(&pts, Grid::<D>::auto_cell(&pts, 4.0));
        for _ in 0..50 {
            let mut lo = [0.0; D];
            let mut hi = [0.0; D];
            let mut q = [0.0; D];
            for i in 0..D {
                let a = rng.f64() * 100.0;
                let w = rng.f64() * 40.0;
                lo[i] = a;
                hi[i] = a + w;
                q[i] = a;
            }
            let b = brute_range(&pts, &lo, &hi);
            assert_eq!(kd.range(&lo, &hi), b);
            assert_eq!(gr.range(&lo, &hi), b);
            let r = rng.f64() * 30.0;
            let br = brute_radius(&pts, &q, r);
            assert_eq!(kd.radius(&q, r), br);
            assert_eq!(gr.radius(&q, r), br);
        }
    }

    #[test]
    fn range_radius_match_brute() {
        for s in 1..6 {
            check_range::<2>(s);
            check_range::<3>(s);
        }
    }

    #[test]
    fn edge_cases() {
        let empty: Vec<Pt<2>> = vec![];
        let kd = KdTree::build(&empty);
        assert!(kd.nearest(&[0.0, 0.0]).is_none());
        let gr = Grid::build(&empty, 1.0);
        assert!(gr.knn(&[0.0, 0.0], 3).is_empty());
        let one = vec![[1.0, 1.0]];
        assert_eq!(KdTree::build(&one).knn(&[0.0, 0.0], 5).len(), 1);
        assert_eq!(Grid::build(&one, 1.0).knn(&[9.0, 9.0], 5).len(), 1);
        // duplicates
        let dup = vec![[2.0, 2.0]; 10];
        assert_eq!(KdTree::build(&dup).knn(&[0.0, 0.0], 4).len(), 4);
        assert_eq!(Grid::build(&dup, 1.0).knn(&[0.0, 0.0], 4).len(), 4);
        assert_eq!(KdTree::build(&dup).knn(&[0.0, 0.0], 0).len(), 0);
    }

    fn rand_boxes(rng: &mut Rng, n: usize) -> Vec<Aabb<2>> {
        (0..n)
            .map(|_| {
                let lo = [rng.f64() * 100.0, rng.f64() * 100.0];
                Aabb::new(lo, [lo[0] + rng.f64() * 12.0, lo[1] + rng.f64() * 12.0])
            })
            .collect()
    }

    #[test]
    fn overlap_sweep_matches_brute() {
        for s in 1..10 {
            let mut rng = Rng::new(s);
            let b = rand_boxes(&mut rng, 150);
            assert_eq!(overlap_pairs_sweep(&b), overlap_pairs_brute(&b));
            let b3: Vec<Aabb<3>> = (0..80)
                .map(|_| {
                    let lo = [rng.f64() * 50.0, rng.f64() * 50.0, rng.f64() * 50.0];
                    Aabb::new(lo, [lo[0] + rng.f64() * 9.0, lo[1] + rng.f64() * 9.0, lo[2] + rng.f64() * 9.0])
                })
                .collect();
            assert_eq!(overlap_pairs_sweep(&b3), overlap_pairs_brute(&b3));
        }
    }

    #[test]
    fn aabb_basics() {
        let a = Aabb::new([0.0, 0.0], [2.0, 2.0]);
        let b = Aabb::new([2.0, 2.0], [3.0, 3.0]);
        let c = Aabb::new([2.1, 0.0], [3.0, 1.0]);
        assert!(a.intersects(&b)); // touching corner
        assert!(!a.intersects(&c));
        assert_eq!(a.intersection_volume(&b), 0.0);
        assert_eq!(a.volume(), 4.0);
        assert_eq!(a.intersection_volume(&Aabb::new([1.0, 1.0], [5.0, 5.0])), 1.0);
    }

    #[test]
    fn astar_matches_bfs() {
        let mut rng = Rng::new(7);
        for t in 0..60 {
            let (w, h) = (12 + rng.below(20), 12 + rng.below(20));
            let s = (rng.below(w), rng.below(h));
            let g = (rng.below(w), rng.below(h));
            let m = GridMap::random(w, h, 0.05 + 0.3 * rng.f64(), &mut rng, &[s, g]);
            let bf = m.bfs_len(s, g);
            let a = m.astar_len(s, g).map(|x| x.0);
            assert_eq!(bf, a, "case {t}");
            if let Some(l) = a {
                assert!(l >= s.0.abs_diff(g.0) + s.1.abs_diff(g.1)); // never shorter than Manhattan
            }
        }
        // open grid: length == Manhattan, A* expands far fewer nodes than the whole grid
        let m = GridMap::new(50, 50);
        let (l, exp) = m.astar_len((0, 0), (49, 49)).unwrap();
        assert_eq!(l, 98);
        assert!(exp <= 50 * 50);
        // unreachable
        let mut m = GridMap::new(5, 5);
        for y in 0..5 {
            m.blocked[y * 5 + 2] = true;
        }
        assert_eq!(m.astar_len((0, 0), (4, 4)), None);
        assert_eq!(m.bfs_len((0, 0), (4, 4)), None);
    }

    #[test]
    fn layout_features_buckets() {
        let sparse: Vec<Aabb<2>> = (0..10).map(|i| Aabb::new([i as f64 * 10.0, 0.0], [i as f64 * 10.0 + 1.0, 1.0])).collect();
        assert_eq!(layout_features(&sparse), vec!["sp:overlap-low".to_string(), "sp:count-few".to_string()]);
        let dense: Vec<Aabb<2>> = (0..30).map(|_| Aabb::new([0.0, 0.0], [5.0, 5.0])).collect();
        assert_eq!(layout_features(&dense)[0], "sp:overlap-high");
    }
}
