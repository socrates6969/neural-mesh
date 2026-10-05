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
        // cell-count cap (G2): a cell far smaller than the data spacing would allocate billions of empty
        // cells; coarsen it until the grid has at most 16 n + 1024 cells (never triggers for sane cells).
        let cap = (16 * pts.len() + 1024) as f64;
        let mut cell = cell;
        let mut dims = [1usize; D];
        loop {
            let mut tot = 1.0f64;
            for i in 0..D {
                dims[i] = (((hi[i] - lo[i]) / cell).floor() as usize + 1).max(1);
                tot *= dims[i] as f64;
            }
            if tot <= cap {
                break;
            }
            cell *= 1.5;
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
    /// Axes along which the data is (numerically) flat are ignored, so a degenerate axis does not
    /// shrink the cell to nothing (G2 fix: a flat axis used to give ~1e-5 cells and billions of cells).
    pub fn auto_cell(pts: &[Pt<D>], per_cell: f64) -> f64 {
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        for p in pts {
            for i in 0..D {
                lo[i] = lo[i].min(p[i]);
                hi[i] = hi[i].max(p[i]);
            }
        }
        if pts.is_empty() {
            return 1.0;
        }
        let widest = (0..D).map(|i| hi[i] - lo[i]).fold(0.0f64, f64::max);
        if widest <= 0.0 {
            return 1.0; // all points identical
        }
        let mut vol = 1.0;
        let mut eff = 0;
        for i in 0..D {
            let w = hi[i] - lo[i];
            if w > widest * 1e-9 {
                vol *= w;
                eff += 1;
            }
        }
        let n = pts.len().max(1) as f64;
        (vol * per_cell / n).powf(1.0 / eff.max(1) as f64).max(widest * 1e-12).max(1e-12)
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
        if ax + 1 == D && !on_edge && r > 0 {
            // last axis and no earlier axis is on the shell: only the two faces c-r and c+r belong to it
            for v in [c[ax] as i64 - r, c[ax] as i64 + r] {
                if v >= lo && v <= hi {
                    cur[ax] = v;
                    self.ring_rec(c, r, ax + 1, cur, true, f);
                }
            }
            return;
        }
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
        let maxr = (0..D).map(|i| c[i].max(self.dims[i] - 1 - c[i])).max().unwrap();
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
// G2 additions (dimension-generic, D = 2, 3, 4 tested; additive, opt-in):
//   * point-to-box distance, nearest / k-NN / intersection queries over a SET OF
//     BOXES (`BoxTree`, a static bounding-volume hierarchy) + brute-force twins,
//   * overlap pairs through the tree (any D, incl. 4-D hyper-boxes = space x time),
//   * `GridN<D>`: BFS / A* on a D-dimensional unit-cost grid with path output,
//   * `SpaceTime<S>`: (S+1)-D space-time planning (S = 3 gives x,y,z,t) with
//     time-monotone moves and waiting, moving obstacles; layered-DP reference.
// ---------------------------------------------------------------------------

impl<const D: usize> Aabb<D> {
    /// Squared Euclidean distance from `q` to the closed box (0 inside / on the surface).
    #[inline]
    pub fn dist2_point(&self, q: &Pt<D>) -> f64 {
        let mut s = 0.0;
        for i in 0..D {
            let d = if q[i] < self.lo[i] {
                self.lo[i] - q[i]
            } else if q[i] > self.hi[i] {
                q[i] - self.hi[i]
            } else {
                0.0
            };
            s += d * d;
        }
        s
    }
    #[inline]
    pub fn contains_point(&self, q: &Pt<D>) -> bool {
        (0..D).all(|i| q[i] >= self.lo[i] && q[i] <= self.hi[i])
    }
}

/// Brute force: `(index, squared distance)` of the box nearest to `q`; ties -> lowest index.
pub fn brute_box_nearest<const D: usize>(boxes: &[Aabb<D>], q: &Pt<D>) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, b) in boxes.iter().enumerate() {
        let d = b.dist2_point(q);
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((i, d));
        }
    }
    best
}

/// Brute force: k nearest boxes `(index, squared distance)`, ascending (ties by index).
pub fn brute_box_knn<const D: usize>(boxes: &[Aabb<D>], q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
    let mut v: Vec<(usize, f64)> = boxes.iter().enumerate().map(|(i, b)| (i, b.dist2_point(q))).collect();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    v.truncate(k);
    v
}

/// Brute force: indices of boxes intersecting the closed `query` box, sorted.
pub fn brute_box_query<const D: usize>(boxes: &[Aabb<D>], query: &Aabb<D>) -> Vec<usize> {
    boxes.iter().enumerate().filter(|(_, b)| b.intersects(query)).map(|(i, _)| i).collect()
}

const BT_NONE: u32 = u32::MAX;
const BT_LEAF: usize = 4;

#[derive(Clone, Debug)]
struct BtNode<const D: usize> {
    lo: Pt<D>,
    hi: Pt<D>,
    left: u32,
    right: u32,
    start: u32,
    end: u32,
}

/// Static bounding-volume hierarchy over boxes (median split of box centres on the
/// widest axis, leaf bucket of <= 4). Boxes must satisfy `lo <= hi` per axis.
pub struct BoxTree<const D: usize> {
    boxes: Vec<Aabb<D>>,
    idx: Vec<u32>,
    nodes: Vec<BtNode<D>>,
}

impl<const D: usize> BoxTree<D> {
    pub fn build(boxes: &[Aabb<D>]) -> Self {
        let bv = boxes.to_vec();
        let mut idx: Vec<u32> = (0..bv.len() as u32).collect();
        let mut nodes = Vec::new();
        if !bv.is_empty() {
            Self::build_rec(&bv, &mut idx[..], 0, &mut nodes);
        }
        BoxTree { boxes: bv, idx, nodes }
    }
    fn build_rec(boxes: &[Aabb<D>], idx: &mut [u32], base: usize, nodes: &mut Vec<BtNode<D>>) -> u32 {
        let mut lo = [f64::INFINITY; D];
        let mut hi = [f64::NEG_INFINITY; D];
        let mut clo = [f64::INFINITY; D];
        let mut chi = [f64::NEG_INFINITY; D];
        for &i in idx.iter() {
            let b = &boxes[i as usize];
            for a in 0..D {
                lo[a] = lo[a].min(b.lo[a]);
                hi[a] = hi[a].max(b.hi[a]);
                let c = 0.5 * (b.lo[a] + b.hi[a]);
                clo[a] = clo[a].min(c);
                chi[a] = chi[a].max(c);
            }
        }
        let id = nodes.len();
        nodes.push(BtNode { lo, hi, left: BT_NONE, right: BT_NONE, start: base as u32, end: (base + idx.len()) as u32 });
        if idx.len() > BT_LEAF {
            let mut ax = 0;
            for a in 1..D {
                if chi[a] - clo[a] > chi[ax] - clo[ax] {
                    ax = a;
                }
            }
            let mid = idx.len() / 2;
            idx.select_nth_unstable_by(mid, |&a, &b| {
                let ca = boxes[a as usize].lo[ax] + boxes[a as usize].hi[ax];
                let cb = boxes[b as usize].lo[ax] + boxes[b as usize].hi[ax];
                ca.partial_cmp(&cb).unwrap_or(Ordering::Equal)
            });
            let (l, r) = idx.split_at_mut(mid);
            let left = Self::build_rec(boxes, l, base, nodes);
            let right = Self::build_rec(boxes, r, base + mid, nodes);
            nodes[id].left = left;
            nodes[id].right = right;
        }
        id as u32
    }
    pub fn len(&self) -> usize {
        self.boxes.len()
    }
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }
    /// Approximate resident bytes.
    pub fn bytes(&self) -> usize {
        self.boxes.len() * D * 16 + self.idx.len() * 4 + self.nodes.len() * (D * 16 + 16)
    }
    #[inline]
    fn node_d2(n: &BtNode<D>, q: &Pt<D>) -> f64 {
        let mut s = 0.0;
        for i in 0..D {
            let d = if q[i] < n.lo[i] {
                n.lo[i] - q[i]
            } else if q[i] > n.hi[i] {
                q[i] - n.hi[i]
            } else {
                0.0
            };
            s += d * d;
        }
        s
    }

    pub fn nearest(&self, q: &Pt<D>) -> Option<(usize, f64)> {
        self.knn(q, 1).into_iter().next()
    }

    /// k nearest boxes `(index, squared distance to the box)`, ascending, ties by index.
    pub fn knn(&self, q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
        if k == 0 || self.boxes.is_empty() {
            return vec![];
        }
        let mut heap: BinaryHeap<(Of, u32)> = BinaryHeap::with_capacity(k + 1);
        self.knn_rec(0, 0.0, q, k, &mut heap);
        let mut v: Vec<(usize, f64)> = heap.into_iter().map(|(d, i)| (i as usize, d.0)).collect();
        v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
        v
    }
    fn knn_rec(&self, n: u32, dmin: f64, q: &Pt<D>, k: usize, heap: &mut BinaryHeap<(Of, u32)>) {
        if heap.len() == k {
            if let Some(&(Of(w), _)) = heap.peek() {
                if dmin > w {
                    return; // strict: equal distance may still win the index tie-break
                }
            }
        }
        let nd = &self.nodes[n as usize];
        if nd.left == BT_NONE {
            for pos in nd.start..nd.end {
                let bi = self.idx[pos as usize];
                let d = self.boxes[bi as usize].dist2_point(q);
                if heap.len() < k {
                    heap.push((Of(d), bi));
                } else if let Some(&(Of(w), wi)) = heap.peek() {
                    if d < w || (d == w && bi < wi) {
                        heap.pop();
                        heap.push((Of(d), bi));
                    }
                }
            }
            return;
        }
        let dl = Self::node_d2(&self.nodes[nd.left as usize], q);
        let dr = Self::node_d2(&self.nodes[nd.right as usize], q);
        if dl <= dr {
            self.knn_rec(nd.left, dl, q, k, heap);
            self.knn_rec(nd.right, dr, q, k, heap);
        } else {
            self.knn_rec(nd.right, dr, q, k, heap);
            self.knn_rec(nd.left, dl, q, k, heap);
        }
    }

    /// Indices of boxes intersecting the closed `query` box (touching counts), sorted.
    pub fn intersecting(&self, query: &Aabb<D>) -> Vec<usize> {
        let mut out = vec![];
        if !self.boxes.is_empty() {
            self.query_rec(0, query, &mut |i| out.push(i));
        }
        out.sort_unstable();
        out
    }
    fn query_rec(&self, n: u32, qb: &Aabb<D>, f: &mut dyn FnMut(usize)) {
        let nd = &self.nodes[n as usize];
        if !(0..D).all(|i| nd.lo[i] <= qb.hi[i] && qb.lo[i] <= nd.hi[i]) {
            return;
        }
        if nd.left == BT_NONE {
            for pos in nd.start..nd.end {
                let bi = self.idx[pos as usize] as usize;
                if self.boxes[bi].intersects(qb) {
                    f(bi);
                }
            }
            return;
        }
        self.query_rec(nd.left, qb, f);
        self.query_rec(nd.right, qb, f);
    }

    /// Boxes containing the point `q` (closed), sorted.
    pub fn containing(&self, q: &Pt<D>) -> Vec<usize> {
        self.intersecting(&Aabb::new(*q, *q))
    }
}

/// All intersecting pairs `(i < j)` through a [`BoxTree`]; identical to
/// [`overlap_pairs_brute`] (sorted). Works for any `D`, e.g. `D = 4` hyper-boxes (x,y,z,t).
pub fn overlap_pairs_tree<const D: usize>(b: &[Aabb<D>]) -> Vec<(usize, usize)> {
    let t = BoxTree::build(b);
    let mut out = vec![];
    if !t.is_empty() {
        for (i, bi) in b.iter().enumerate() {
            t.query_rec(0, bi, &mut |j| {
                if j > i {
                    out.push((i, j));
                }
            });
        }
    }
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// D-dimensional unit-cost grid path finding
// ---------------------------------------------------------------------------

/// A path-finding result.
#[derive(Clone, Debug, PartialEq)]
pub struct PathN<const D: usize> {
    /// number of unit steps
    pub len: usize,
    /// nodes popped (and not stale) by the search
    pub expanded: usize,
    /// start..=goal cell coordinates
    pub path: Vec<[usize; D]>,
}

#[derive(Clone, Debug)]
pub struct GridN<const D: usize> {
    pub dims: [usize; D],
    pub blocked: Vec<bool>,
}

impl<const D: usize> GridN<D> {
    pub fn new(dims: [usize; D]) -> Self {
        let total: usize = dims.iter().product();
        GridN { dims, blocked: vec![false; total] }
    }
    pub fn cells(&self) -> usize {
        self.blocked.len()
    }
    /// Random obstacles with the given density, never blocking `keep_free` cells.
    pub fn random(dims: [usize; D], density: f64, rng: &mut crate::router::Rng, keep_free: &[[usize; D]]) -> Self {
        let mut m = GridN::new(dims);
        for i in 0..m.cells() {
            m.blocked[i] = rng.bernoulli(density);
        }
        for c in keep_free {
            let f = m.flat(c);
            m.blocked[f] = false;
        }
        m
    }
    #[inline]
    pub fn flat(&self, c: &[usize; D]) -> usize {
        let mut f = 0;
        for i in 0..D {
            f = f * self.dims[i] + c[i];
        }
        f
    }
    #[inline]
    pub fn coords(&self, mut f: usize) -> [usize; D] {
        let mut c = [0usize; D];
        for i in (0..D).rev() {
            c[i] = f % self.dims[i];
            f /= self.dims[i];
        }
        c
    }
    #[inline]
    pub fn in_bounds(&self, c: &[usize; D]) -> bool {
        (0..D).all(|i| c[i] < self.dims[i])
    }
    #[inline]
    pub fn is_open(&self, c: &[usize; D]) -> bool {
        self.in_bounds(c) && !self.blocked[self.flat(c)]
    }
    #[inline]
    fn for_neighbours(&self, c: &[usize; D], f: &mut dyn FnMut([usize; D])) {
        for ax in 0..D {
            if c[ax] > 0 {
                let mut n = *c;
                n[ax] -= 1;
                if !self.blocked[self.flat(&n)] {
                    f(n);
                }
            }
            if c[ax] + 1 < self.dims[ax] {
                let mut n = *c;
                n[ax] += 1;
                if !self.blocked[self.flat(&n)] {
                    f(n);
                }
            }
        }
    }
    /// Shortest axis-aligned (2*D neighbours, unit cost) path length by BFS; the reference.
    pub fn bfs_len(&self, s: [usize; D], g: [usize; D]) -> Option<usize> {
        if !self.is_open(&s) || !self.is_open(&g) {
            return None;
        }
        let mut dist = vec![usize::MAX; self.cells()];
        let mut dq = VecDeque::new();
        dist[self.flat(&s)] = 0;
        dq.push_back(s);
        while let Some(c) = dq.pop_front() {
            let d = dist[self.flat(&c)];
            if c == g {
                return Some(d);
            }
            let mut nb: Vec<[usize; D]> = Vec::with_capacity(2 * D);
            self.for_neighbours(&c, &mut |n| nb.push(n));
            for n in nb {
                let f = self.flat(&n);
                if dist[f] == usize::MAX {
                    dist[f] = d + 1;
                    dq.push_back(n);
                }
            }
        }
        None
    }
    /// A* with the Manhattan heuristic (admissible and consistent), returning the path.
    pub fn astar(&self, s: [usize; D], g: [usize; D]) -> Option<PathN<D>> {
        if !self.is_open(&s) || !self.is_open(&g) {
            return None;
        }
        let h = |c: &[usize; D]| -> usize { (0..D).map(|i| c[i].abs_diff(g[i])).sum() };
        let mut best = vec![usize::MAX; self.cells()];
        let mut parent = vec![u32::MAX; self.cells()];
        let mut heap: BinaryHeap<std::cmp::Reverse<(usize, usize, u32)>> = BinaryHeap::new();
        let sf = self.flat(&s);
        best[sf] = 0;
        heap.push(std::cmp::Reverse((h(&s), 0, sf as u32)));
        let mut expanded = 0;
        while let Some(std::cmp::Reverse((_f, gc, cf))) = heap.pop() {
            if gc > best[cf as usize] {
                continue;
            }
            expanded += 1;
            let c = self.coords(cf as usize);
            if c == g {
                let mut path = vec![c];
                let mut cur = cf;
                while parent[cur as usize] != u32::MAX {
                    cur = parent[cur as usize];
                    path.push(self.coords(cur as usize));
                }
                path.reverse();
                return Some(PathN { len: gc, expanded, path });
            }
            let mut nb: Vec<[usize; D]> = Vec::with_capacity(2 * D);
            self.for_neighbours(&c, &mut |n| nb.push(n));
            for n in nb {
                let nf = self.flat(&n);
                let ng = gc + 1;
                if ng < best[nf] {
                    best[nf] = ng;
                    parent[nf] = cf;
                    heap.push(std::cmp::Reverse((ng + h(&n), ng, nf as u32)));
                }
            }
        }
        None
    }
    /// `true` iff `p` is a valid unit-step open-cell path from `s` to `g`.
    pub fn path_is_valid(&self, p: &[[usize; D]], s: [usize; D], g: [usize; D]) -> bool {
        if p.is_empty() || p[0] != s || *p.last().unwrap() != g {
            return false;
        }
        p.iter().all(|c| self.is_open(c)) && p.windows(2).all(|w| (0..D).map(|i| w[0][i].abs_diff(w[1][i])).sum::<usize>() == 1)
    }
}

// ---------------------------------------------------------------------------
// Space-time planning: (S spatial dims + time). S = 3 is the 4-D (x,y,z,t) case.
// ---------------------------------------------------------------------------

/// Result of a space-time search.
#[derive(Clone, Debug, PartialEq)]
pub struct SpaceTimePath<const S: usize> {
    /// total cost = moves * move_cost + waits * wait_cost
    pub cost: u64,
    /// time step at which the goal cell is first reached on this path
    pub arrival: usize,
    /// states popped by the search
    pub expanded: usize,
    /// `(cell, t)` for t = 0..=arrival (waiting repeats the cell)
    pub path: Vec<([usize; S], usize)>,
}

/// Space-time grid: a state is `(cell, t)`; every step advances `t` by exactly 1 and either
/// waits (stay) or moves one cell along one axis (time-monotone). A cell can be blocked for
/// all time (static) or only at certain steps (moving obstacle, vertex-conflict semantics:
/// the agent may not occupy `(cell, t)` if it is flagged; swaps are not checked).
#[derive(Clone, Debug)]
pub struct SpaceTime<const S: usize> {
    pub dims: [usize; S],
    /// last time step that may be used (states have `t` in `0..=horizon`)
    pub horizon: usize,
    pub move_cost: u32,
    pub wait_cost: u32,
    occupied: Vec<bool>, // [t * cells + cell]
}

impl<const S: usize> SpaceTime<S> {
    pub fn new(dims: [usize; S], horizon: usize) -> Self {
        let cells: usize = dims.iter().product();
        SpaceTime { dims, horizon, move_cost: 1, wait_cost: 1, occupied: vec![false; cells * (horizon + 1)] }
    }
    pub fn cells(&self) -> usize {
        self.dims.iter().product()
    }
    #[inline]
    pub fn flat(&self, c: &[usize; S]) -> usize {
        let mut f = 0;
        for i in 0..S {
            f = f * self.dims[i] + c[i];
        }
        f
    }
    #[inline]
    pub fn coords(&self, mut f: usize) -> [usize; S] {
        let mut c = [0usize; S];
        for i in (0..S).rev() {
            c[i] = f % self.dims[i];
            f /= self.dims[i];
        }
        c
    }
    pub fn block_static(&mut self, c: &[usize; S]) {
        let f = self.flat(c);
        let n = self.cells();
        for t in 0..=self.horizon {
            self.occupied[t * n + f] = true;
        }
    }
    pub fn block_at(&mut self, c: &[usize; S], t: usize) {
        let f = self.flat(c);
        let n = self.cells();
        self.occupied[t * n + f] = true;
    }
    #[inline]
    pub fn is_free(&self, c: &[usize; S], t: usize) -> bool {
        t <= self.horizon && (0..S).all(|i| c[i] < self.dims[i]) && !self.occupied[t * self.cells() + self.flat(c)]
    }
    /// Random world: static obstacles with density `static_density`, plus each `(cell, t)` blocked
    /// independently with probability `dyn_density`; the `(cell, t)` pairs in `keep_free` are cleared.
    pub fn random(
        dims: [usize; S],
        horizon: usize,
        static_density: f64,
        dyn_density: f64,
        rng: &mut crate::router::Rng,
        keep_free: &[([usize; S], usize)],
    ) -> Self {
        let mut w = SpaceTime::new(dims, horizon);
        let n = w.cells();
        for f in 0..n {
            if rng.bernoulli(static_density) {
                for t in 0..=horizon {
                    w.occupied[t * n + f] = true;
                }
            }
        }
        for i in 0..w.occupied.len() {
            if rng.bernoulli(dyn_density) {
                w.occupied[i] = true;
            }
        }
        for (c, t) in keep_free {
            let f = w.flat(c);
            w.occupied[*t * n + f] = false;
        }
        w
    }
    #[inline]
    fn manhattan_to(c: &[usize; S], g: &[usize; S]) -> u64 {
        (0..S).map(|i| c[i].abs_diff(g[i]) as u64).sum()
    }
    #[inline]
    fn for_moves(&self, c: &[usize; S], f: &mut dyn FnMut([usize; S], u32)) {
        f(*c, self.wait_cost);
        for ax in 0..S {
            if c[ax] > 0 {
                let mut n = *c;
                n[ax] -= 1;
                f(n, self.move_cost);
            }
            if c[ax] + 1 < self.dims[ax] {
                let mut n = *c;
                n[ax] += 1;
                f(n, self.move_cost);
            }
        }
    }

    /// Minimum-cost schedule from `(s, 0)` to the first time the agent stands on `g`
    /// (any `t <= horizon`), by A* over `(cell, t)` with h = Manhattan * move_cost
    /// (admissible and consistent: each unit of Manhattan distance needs a move).
    pub fn astar(&self, s: [usize; S], g: [usize; S]) -> Option<SpaceTimePath<S>> {
        if !self.is_free(&s, 0) {
            return None;
        }
        let n = self.cells();
        let total = n * (self.horizon + 1);
        let mut best = vec![u64::MAX; total];
        let mut parent = vec![u32::MAX; total];
        let mc = self.move_cost as u64;
        let mut heap: BinaryHeap<std::cmp::Reverse<(u64, u64, u32)>> = BinaryHeap::new();
        let s0 = self.flat(&s);
        best[s0] = 0;
        heap.push(std::cmp::Reverse((Self::manhattan_to(&s, &g) * mc, 0, s0 as u32)));
        let mut expanded = 0;
        while let Some(std::cmp::Reverse((_f, gc, st))) = heap.pop() {
            if gc > best[st as usize] {
                continue;
            }
            expanded += 1;
            let t = st as usize / n;
            let c = self.coords(st as usize % n);
            if c == g {
                let mut path = vec![(c, t)];
                let mut cur = st;
                while parent[cur as usize] != u32::MAX {
                    cur = parent[cur as usize];
                    path.push((self.coords(cur as usize % n), cur as usize / n));
                }
                path.reverse();
                return Some(SpaceTimePath { cost: gc, arrival: t, expanded, path });
            }
            if t == self.horizon {
                continue;
            }
            let mut nb: Vec<([usize; S], u32)> = Vec::with_capacity(2 * S + 1);
            self.for_moves(&c, &mut |m, w| nb.push((m, w)));
            for (m, w) in nb {
                if !self.is_free(&m, t + 1) {
                    continue;
                }
                let ns = ((t + 1) * n + self.flat(&m)) as u32;
                let ng = gc + w as u64;
                if ng < best[ns as usize] {
                    best[ns as usize] = ng;
                    parent[ns as usize] = st;
                    heap.push(std::cmp::Reverse((ng + Self::manhattan_to(&m, &g) * mc, ng, ns)));
                }
            }
        }
        None
    }

    /// Reference: exact layered dynamic programme over time (no heuristic, no heap). Returns the
    /// minimum cost to be standing on `g` at any `t <= horizon` and the earliest `t` achieving it.
    pub fn dp_min_cost(&self, s: [usize; S], g: [usize; S]) -> Option<(u64, usize)> {
        if !self.is_free(&s, 0) {
            return None;
        }
        let n = self.cells();
        let mut cur = vec![u64::MAX; n];
        cur[self.flat(&s)] = 0;
        let gf = self.flat(&g);
        let mut best: Option<(u64, usize)> = None;
        for t in 0..=self.horizon {
            if cur[gf] != u64::MAX && best.map_or(true, |(b, _)| cur[gf] < b) {
                best = Some((cur[gf], t));
            }
            if t == self.horizon {
                break;
            }
            let mut next = vec![u64::MAX; n];
            for f in 0..n {
                if cur[f] == u64::MAX {
                    continue;
                }
                let c = self.coords(f);
                self.for_moves(&c, &mut |m, w| {
                    if self.is_free(&m, t + 1) {
                        let mf = self.flat(&m);
                        let v = cur[f] + w as u64;
                        if v < next[mf] {
                            next[mf] = v;
                        }
                    }
                });
            }
            cur = next;
        }
        best
    }

    /// Checks a schedule: starts at `(s,0)`, ends on `g`, `t` increases by exactly 1 per step, every
    /// step is a wait or a unit axis move into a free `(cell, t)`, and the summed cost equals `cost`.
    pub fn path_is_valid(&self, p: &[([usize; S], usize)], s: [usize; S], g: [usize; S], cost: u64) -> bool {
        if p.is_empty() || p[0] != (s, 0) || p.last().unwrap().0 != g || !self.is_free(&p[0].0, 0) {
            return false;
        }
        let mut total = 0u64;
        for w in p.windows(2) {
            let (a, ta) = w[0];
            let (b, tb) = w[1];
            if tb != ta + 1 || !self.is_free(&b, tb) {
                return false;
            }
            let d: usize = (0..S).map(|i| a[i].abs_diff(b[i])).sum();
            match d {
                0 => total += self.wait_cost as u64,
                1 => total += self.move_cost as u64,
                _ => return false,
            }
        }
        total == cost
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

    // ------------------------------------------------------------------ G2: D = 2, 3, 4

    #[derive(Clone, Copy, Debug)]
    enum Gen {
        Uniform,
        Lattice,
        Duplicates,
        Clustered,
        Identical,
        Flat,
    }

    fn gen_pts<const D: usize>(rng: &mut Rng, n: usize, g: Gen) -> Vec<Pt<D>> {
        let base: Vec<f64> = (0..D).map(|_| (rng.f64() * 5.0).floor()).collect();
        (0..n)
            .map(|_| {
                let mut p = [0.0; D];
                for i in 0..D {
                    p[i] = match g {
                        Gen::Uniform => rng.f64() * 100.0,
                        Gen::Lattice => (rng.f64() * 6.0).floor(),
                        Gen::Duplicates => (rng.f64() * 3.0).floor() * 7.5,
                        Gen::Clustered => (rng.f64() * 8.0).floor() + rng.f64() * 0.01,
                        Gen::Identical => base[i],
                        Gen::Flat => {
                            if i == 0 {
                                3.0
                            } else {
                                rng.f64() * 10.0
                            }
                        }
                    };
                }
                p
            })
            .collect()
    }

    /// Exact (index AND squared distance) comparison, including ties.
    fn check_points_exact<const D: usize>(seed: u64, g: Gen, n: usize) -> usize {
        let mut rng = Rng::new(seed);
        let pts: Vec<Pt<D>> = gen_pts(&mut rng, n, g);
        let kd = KdTree::build(&pts);
        let grid = Grid::build(&pts, Grid::<D>::auto_cell(&pts, 3.0));
        let mut bad = 0;
        for _ in 0..40 {
            let mut q = [0.0; D];
            for i in 0..D {
                q[i] = match g {
                    Gen::Uniform | Gen::Clustered => rng.f64() * 110.0 - 5.0,
                    _ => (rng.f64() * 8.0).floor() - 1.0,
                };
            }
            let k = rng.below(n + 4); // includes 0 and k > n
            let b = brute_knn(&pts, &q, k);
            if kd.knn(&q, k) != b {
                bad += 1;
            }
            if grid.knn(&q, k) != b {
                bad += 1;
            }
            if kd.nearest(&q) != brute_nearest(&pts, &q) || grid.nearest(&q) != brute_nearest(&pts, &q) {
                bad += 1;
            }
            let mut lo = q;
            let mut hi = q;
            for i in 0..D {
                let w = (rng.f64() * 4.0).floor();
                lo[i] -= w;
                hi[i] += w + (rng.below(2) as f64) * 0.5;
            }
            let br = brute_range(&pts, &lo, &hi);
            if kd.range(&lo, &hi) != br || grid.range(&lo, &hi) != br {
                bad += 1;
            }
            let r = (rng.f64() * 4.0).floor() + (rng.below(2) as f64) * 0.5;
            let rb = brute_radius(&pts, &q, r);
            if kd.radius(&q, r) != rb || grid.radius(&q, r) != rb {
                bad += 1;
            }
        }
        bad
    }

    #[test]
    fn g2_points_2d_3d_4d_exact_incl_ties_and_degenerate() {
        let mut bad = 0;
        for s in 1..=12u64 {
            for g in [Gen::Uniform, Gen::Lattice, Gen::Duplicates, Gen::Clustered, Gen::Identical, Gen::Flat] {
                for n in [0usize, 1, 2, 7, 60, 200] {
                    bad += check_points_exact::<2>(s * 31 + n as u64, g, n);
                    bad += check_points_exact::<3>(s * 37 + n as u64, g, n);
                    bad += check_points_exact::<4>(s * 41 + n as u64, g, n);
                }
            }
        }
        assert_eq!(bad, 0, "kd/grid vs brute mismatches in 2D/3D/4D");
    }

    #[test]
    fn g2_points_4d_uniform_matches_brute() {
        for s in 1..4 {
            check_knn::<4>(s, false);
            check_knn::<4>(s + 100, true);
            check_range::<4>(s);
        }
    }

    #[test]
    fn g2_inverted_range_is_empty() {
        let mut rng = Rng::new(3);
        let pts: Vec<Pt<3>> = gen_pts(&mut rng, 100, Gen::Uniform);
        let kd = KdTree::build(&pts);
        let gr = Grid::build(&pts, 10.0);
        let (lo, hi) = ([60.0, 60.0, 60.0], [20.0, 20.0, 20.0]);
        assert!(brute_range(&pts, &lo, &hi).is_empty());
        assert!(kd.range(&lo, &hi).is_empty());
        assert!(gr.range(&lo, &hi).is_empty());
    }

    fn gen_boxes<const D: usize>(rng: &mut Rng, n: usize, kind: u8) -> Vec<Aabb<D>> {
        (0..n)
            .map(|_| {
                let mut lo = [0.0; D];
                let mut hi = [0.0; D];
                for i in 0..D {
                    match kind {
                        // continuous, mixed sizes
                        0 => {
                            lo[i] = rng.f64() * 100.0;
                            hi[i] = lo[i] + rng.f64() * 14.0;
                        }
                        // integer lattice: lots of exactly-touching faces
                        1 => {
                            lo[i] = (rng.f64() * 8.0).floor();
                            hi[i] = lo[i] + (rng.f64() * 3.0).floor();
                        }
                        // zero-extent (points) on the lattice
                        2 => {
                            lo[i] = (rng.f64() * 5.0).floor();
                            hi[i] = lo[i];
                        }
                        // one huge slab mixed with tiny boxes
                        _ => {
                            if rng.below(10) == 0 {
                                lo[i] = 0.0;
                                hi[i] = 100.0;
                            } else {
                                lo[i] = rng.f64() * 100.0;
                                hi[i] = lo[i] + 0.5;
                            }
                        }
                    }
                }
                Aabb::new(lo, hi)
            })
            .collect()
    }

    fn check_boxes<const D: usize>(seed: u64, kind: u8, n: usize) -> usize {
        let mut rng = Rng::new(seed);
        let boxes: Vec<Aabb<D>> = gen_boxes(&mut rng, n, kind);
        let t = BoxTree::build(&boxes);
        let mut bad = 0;
        for _ in 0..30 {
            let mut q = [0.0; D];
            for i in 0..D {
                q[i] = if kind == 0 || kind == 3 { rng.f64() * 120.0 - 10.0 } else { (rng.f64() * 10.0).floor() - 1.0 };
            }
            let k = rng.below(n + 3);
            if t.knn(&q, k) != brute_box_knn(&boxes, &q, k) {
                bad += 1;
            }
            if t.nearest(&q) != brute_box_nearest(&boxes, &q) {
                bad += 1;
            }
            let qb = gen_boxes::<D>(&mut rng, 1, kind)[0];
            if t.intersecting(&qb) != brute_box_query(&boxes, &qb) {
                bad += 1;
            }
            let pb = Aabb::new(q, q);
            if t.containing(&q) != brute_box_query(&boxes, &pb) {
                bad += 1;
            }
        }
        if overlap_pairs_tree(&boxes) != overlap_pairs_brute(&boxes) || overlap_pairs_sweep(&boxes) != overlap_pairs_brute(&boxes) {
            bad += 1;
        }
        bad
    }

    #[test]
    fn g2_box_tree_and_overlap_2d_3d_4d_exact() {
        let mut bad = 0;
        for s in 1..=10u64 {
            for kind in 0..4u8 {
                for n in [0usize, 1, 3, 20, 120] {
                    bad += check_boxes::<2>(s * 13 + n as u64, kind, n);
                    bad += check_boxes::<3>(s * 17 + n as u64, kind, n);
                    bad += check_boxes::<4>(s * 19 + n as u64, kind, n);
                }
            }
        }
        assert_eq!(bad, 0, "BoxTree/overlap vs brute mismatches");
    }

    #[test]
    fn g2_hyperbox_touching_in_time_only() {
        // two 3-D unit cubes at the same place but disjoint in time; touching in time counts
        let a = Aabb::new([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 5.0]);
        let b = Aabb::new([0.5, 0.5, 0.5, 5.0], [2.0, 2.0, 2.0, 9.0]); // t touches at 5
        let c = Aabb::new([0.5, 0.5, 0.5, 5.0001], [2.0, 2.0, 2.0, 9.0]);
        assert!(a.intersects(&b) && !a.intersects(&c));
        assert_eq!(overlap_pairs_tree(&[a, b, c]), vec![(0, 1), (1, 2)]);
        assert_eq!(a.dist2_point(&[2.0, 0.5, 0.5, 2.0]), 1.0);
        assert_eq!(a.dist2_point(&[0.5, 0.5, 0.5, 2.0]), 0.0);
    }

    fn check_gridn<const D: usize>(seed: u64, dim: usize) -> usize {
        let mut rng = Rng::new(seed);
        let mut dims = [dim; D];
        for i in 0..D {
            dims[i] = (dim + rng.below(4)).max(2);
        }
        let mut s = [0usize; D];
        let mut g = [0usize; D];
        for i in 0..D {
            s[i] = rng.below(dims[i]);
            g[i] = rng.below(dims[i]);
        }
        let m = GridN::random(dims, 0.05 + 0.4 * rng.f64(), &mut rng, &[s, g]);
        let bf = m.bfs_len(s, g);
        let a = m.astar(s, g);
        let mut bad = 0;
        if bf != a.as_ref().map(|p| p.len) {
            bad += 1;
        }
        if let Some(p) = a {
            if !m.path_is_valid(&p.path, s, g) || p.path.len() != p.len + 1 {
                bad += 1;
            }
            let man: usize = (0..D).map(|i| s[i].abs_diff(g[i])).sum();
            if p.len < man {
                bad += 1;
            }
        }
        bad
    }

    #[test]
    fn g2_gridn_astar_matches_bfs_3d_4d() {
        let mut bad = 0;
        for s in 0..120u64 {
            bad += check_gridn::<2>(s, 14);
            bad += check_gridn::<3>(s + 1000, 7);
            bad += check_gridn::<4>(s + 2000, 4);
        }
        assert_eq!(bad, 0);
        // open 3-D grid: path length == Manhattan, endpoints & blocked cases
        let m = GridN::<3>::new([10, 10, 10]);
        let p = m.astar([0, 0, 0], [9, 9, 9]).unwrap();
        assert_eq!(p.len, 27);
        assert_eq!(m.astar([1, 1, 1], [1, 1, 1]).unwrap().len, 0);
        let mut w = GridN::<3>::new([5, 5, 5]);
        for y in 0..5 {
            for z in 0..5 {
                let f = w.flat(&[2, y, z]);
                w.blocked[f] = true;
            }
        }
        assert!(w.astar([0, 0, 0], [4, 4, 4]).is_none() && w.bfs_len([0, 0, 0], [4, 4, 4]).is_none());
        assert!(w.astar([2, 0, 0], [4, 4, 4]).is_none()); // blocked start
        assert!(w.astar([0, 0, 0], [5, 0, 0]).is_none()); // out of bounds goal
    }

    fn check_spacetime<const S: usize>(seed: u64, side: usize, horizon: usize) -> usize {
        let mut rng = Rng::new(seed);
        let mut dims = [side; S];
        for i in 0..S {
            dims[i] = (side + rng.below(3)).max(2);
        }
        let mut s = [0usize; S];
        let mut g = [0usize; S];
        for i in 0..S {
            s[i] = rng.below(dims[i]);
            g[i] = rng.below(dims[i]);
        }
        let mut w = SpaceTime::random(dims, horizon, 0.1 * rng.f64(), 0.05 + 0.3 * rng.f64(), &mut rng, &[(s, 0)]);
        w.move_cost = 1 + rng.below(3) as u32;
        w.wait_cost = rng.below(3) as u32; // includes free waiting
        let a = w.astar(s, g);
        let d = w.dp_min_cost(s, g);
        let mut bad = 0;
        match (&a, d) {
            (None, None) => {}
            (Some(p), Some((c, _))) => {
                if p.cost != c || !w.path_is_valid(&p.path, s, g, p.cost) || p.path.last().unwrap().1 != p.arrival {
                    bad += 1;
                }
                // first-arrival property: goal not visited earlier on the path
                if p.path[..p.path.len() - 1].iter().any(|(c, _)| *c == g) {
                    bad += 1;
                }
            }
            _ => bad += 1,
        }
        bad
    }

    #[test]
    fn g2_spacetime_astar_matches_layered_dp_3d_4d() {
        let mut bad = 0;
        for s in 0..150u64 {
            bad += check_spacetime::<2>(s, 6, 14); // (x,y,t): 3-D
            bad += check_spacetime::<3>(s + 500, 4, 12); // (x,y,z,t): 4-D
        }
        assert_eq!(bad, 0);
    }

    #[test]
    fn g2_spacetime_waiting_and_monotone_time() {
        // 1-D corridor of 5 cells, goal at the far end; a mover blocks cell 2 at t = 2 and 3.
        let mut w = SpaceTime::<1>::new([5], 10);
        w.block_at(&[2], 2);
        w.block_at(&[2], 3);
        let p = w.astar([0], [4]).unwrap();
        // direct trip would enter cell 2 at t=2 (blocked): must wait. Best arrival is t=6 (4 moves + 2 waits)?
        // enter cell 2 at t=4 earliest after moving 0->1 at t=1 and waiting at 1 for t=2,3 -> arrival 4+... verify by DP.
        assert_eq!(Some((p.cost, p.arrival)), w.dp_min_cost([0], [4]));
        assert!(p.cost > 4 && p.arrival > 4);
        assert!(w.path_is_valid(&p.path, [0], [4], p.cost));
        // t strictly increases by one per entry
        assert!(p.path.windows(2).all(|x| x[1].1 == x[0].1 + 1));
        // horizon too short -> no path
        let mut short = SpaceTime::<1>::new([5], 4);
        short.block_at(&[2], 2);
        assert!(short.astar([0], [4]).is_none() && short.dp_min_cost([0], [4]).is_none());
        // fully blocked static cell on a 1-D corridor
        let mut wall = SpaceTime::<1>::new([5], 20);
        wall.block_static(&[2]);
        assert!(wall.astar([0], [4]).is_none());
        // start == goal costs 0; blocked start -> None
        assert_eq!(SpaceTime::<2>::new([3, 3], 3).astar([1, 1], [1, 1]).unwrap().cost, 0);
        let mut bs = SpaceTime::<2>::new([3, 3], 3);
        bs.block_at(&[0, 0], 0);
        assert!(bs.astar([0, 0], [2, 2]).is_none());
        // 4-D open world: cost == Manhattan moves, arrival == Manhattan
        let o = SpaceTime::<3>::new([5, 5, 5], 20);
        let r = o.astar([0, 0, 0], [4, 3, 2]).unwrap();
        assert_eq!((r.cost, r.arrival), (9, 9));
    }
}
