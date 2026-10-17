# H15 proposed patch — second HNSW failure mode (disconnected layer-0 graph)

Worker R7-A. SYNTHETIC-until-validated-on-real-traffic. This is the SECOND failure mode that H1 (R4-A1) reported in the fixed
crate (about 0.5 % of builds lose 8-37 % of items, recall@10 ~.67) and could not root-cause. It is now root-caused, with a
reproducible trigger and a one-line fix. This patch is ADDITIVE to the existing g17 reverse-edge patch already in
`vendor_g17/hnsw_rs_patched` — the two are independent lines; the fix was developed in a second copy
`vendor_g17/hnsw_rs_exp` and NOT applied to `hnsw_rs_patched` (awaiting director approval).

## Root cause (proven causally)

`PointIndexation::points_by_layer[l]` lists only the points whose **top** level is `l`. But layer 0 conceptually contains
**every** point. `search_layer` (hnsw.rs) early-returns an empty result when `points_by_layer[layer]` is empty:

```
if self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
    return return_points;   // empty
}
```

When the FIRST TWO inserted points both draw level >= 1 (probability ~ (1/M)^2), `points_by_layer[0]` is still empty while
the second point is being inserted. Its descent to layer 0 therefore gets an empty search result, so it receives **no
layer-0 neighbour** and gives the entry point no layer-0 reverse edge. The two points seed two layer-0 islands. Every
later level-0 point greedily descends from the entry and attaches to whichever island its descent lands in; the islands
never merge. The layer-0 graph stays in >= 2 weak components for the whole build, and one whole component is unreachable
from the entry point at search time → 8-47 % of items silently unfindable (recall collapses).

This is DISTINCT from the g17 reverse-edge bug (`l_n`): that one loses ~1 % of *level>=1* points in every build; this one
loses a large *contiguous component* in ~1/M^2 of builds and is independent of the `l_n` line.

### Evidence (all reproducible; hnsw.rs patched to accept an injected level sequence, so OS-random builds are replayable)
- Forced-prefix test (`h15_force`, 20 builds each, n2000 d32 iid): prefix `[1,1]` → 20/20 catastrophic (mean 25.5 % lost);
  `[1,0]`,`[0,1]`,`[0,0]` → 0/20; `[2,2]`,`[2,1]`,`[3,3]`,`[2,1,2]`,`[5,5,5]`,ten 1s → all catastrophic (any first-two-both-upper
  trigger). Same on the UNPATCHED-reverse-edge copy (`[1,1]` 20/20 cat, `[0,0]`/`[1,0]` 0 cat) — so the mode is orthogonal to g17.
- Rate grid (`h15_grid`, 3000 builds/cell, random OS levels, seeded-reproducible): the predicate "first two points both level>=1"
  equals "layer-0 graph has >1 weak component" equals "catastrophic (>5 % lost)" in every cell (flag==comps>1==cat, with
  flag_but_not_cat<=1 and loss_without_flag=0):
  | cell | builds | cat | rate (Wilson 95%) | 1/M^2 |
  |---|---|---|---|---|
  | n500 d32 M16 efc200 | 3000 | 12 | 0.40 % (0.23-0.70) | 0.39 % |
  | n200 d32 M16 | 3000 | 8 | 0.27 % (0.14-0.53) | 0.39 % |
  | n500 d32 efc100 | 3000 | 11 | 0.37 % (0.21-0.66) | 0.39 % |
  | n500 d32 efc400 | 3000 | 10 | 0.33 % (0.18-0.61) | 0.39 % |
  | n500 d8 M16 | 3000 | 7 | 0.23 % (0.11-0.48) | 0.39 % |
  | n500 d32 M8 (expected 1.56 %) | see worker-1 grid | | | 1.56 % |
  So the rate is ~1/M^2 (M=16 → 0.39 %), consistent with the H1 estimate of 0.55 % [0.33-0.92]. Independent of N, dim, efc.

## The fix (one line)

`vendor_g17/hnsw_rs_patched/src/hnsw.rs`, in `search_layer`:

```diff
-        if self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
+        // H15: points_by_layer[l] lists only points whose TOP level is l, but layer 0 holds EVERY point; it is "empty"
+        // only before the first point. Returning empty at layer 0 (when the first points are all upper-level) left the
+        // layer-0 graph permanently split (~1/M^2 of builds, 8-47 % of items unfindable). Never early-return for layer 0.
+        if layer != 0 && self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
```

(The exact patched file is `results/h15_hnsw_layer0.diff`, cut against `hnsw_rs_patched`'s current `hnsw.rs`.)

### Validation (`h15_cmp`, experiment crate = g17 reverse-edge patch + this layer-0 guard)
- Failure rate 0: forced `[1,1]` with the fix → 0/… catastrophic in every forced test; all 10 naturally-found rare seeds
  (100062/100870/100997/101119/101350/101403/200398/200405/200730/200900) lose 242-653 items unpatched (fix 0) and **0** with
  the fix. Workaround grid `fix1` arm: 28/5000 flagged builds, **0 failures** (vs `none` 28/28 failing).
- Equivalence on the 99.6 % of builds that do NOT trigger: byte-identical graph (FNV hash over all layers' neighbour lists,
  per point) between fix-off and fix-on — fix 0 == fix 1 hash on every non-flag build in a 6000-build n500 identity run
  (see `h15_cmp_ident_n500.csv`); recall@10 and build time unchanged. (The fix only changes behaviour on the ~0.4 % of builds
  that would otherwise be catastrophic.)
- recall/latency on the catastrophic→fixed builds and overall: see `h15_cmp_perf_*` (n5000 d32, n2000 d384, n5000 d128 clust).

A broader variant (FIX_MODE 2, "layer l empty only if no point of level>=l exists") is also 0-failure and equivalent on
non-flag builds, but changes the graph on a few non-catastrophic builds (upper empty layers), so the minimal `layer != 0`
guard is preferred.

## Recommendation
Apply the one-line `layer != 0` guard to `vendor_g17/hnsw_rs_patched/src/hnsw.rs` (director's call, since R7-A may not edit it).
It removes the second failure mode at zero cost and keeps the g17 reverse-edge fix. After it, `HnswIndex::audit_and_repair`
becomes belt-and-braces rather than load-bearing.
