# H15 (R7-A) — root-cause of the rare catastrophic HNSW build in the fixed crate

All numbers SYNTHETIC (iid Gaussian / N-100-cluster / exact-duplicate / low-intrinsic-dim), until validated on real traffic.
Box shared/loaded: build-ms and us/query are indicative. No defaults changed; nothing in the repo `Cargo.toml` / `src/*.rs` /
`~/.cargo/registry` touched. All experiment code is a STANDALONE cargo project `vendor_g17/harness2` (+ `harness2u`) depending on a
SECOND vendored copy `vendor_g17/hnsw_rs_exp` (`hnsw_rs_exp_unp` = reverse-edge-UNpatched twin); `vendor_g17/hnsw_rs_patched`
left exactly as the director set it.

## VERIFY REQUEST — R7-A — H15 second HNSW failure mode: ROOT-CAUSED + one-line fix

RESULT (plain): the rare catastrophic build H1 reported (fixed crate, ~0.5 % of builds lose 8-37 %, recall@10 ~.67, "not
root-caused") is now root-caused with a reproducible trigger, a one-line fix, and a quantified rate. It is a SECOND, distinct
bug from the g17 reverse-edge one. **It fires in exactly 1/M^2 of builds** (the first two inserted points both drawing level>=1),
splitting the layer-0 graph into >=2 permanent components so one whole component is unreachable from the entry point.

### Root cause
`PointIndexation::points_by_layer[l]` lists only points whose **top** level is `l`, but layer 0 holds every point. `search_layer`
early-returns empty when `points_by_layer[layer].is_empty()`. If the first two inserts both get level>=1, `points_by_layer[0]`
is still empty while point #2 descends to layer 0 → it gets no layer-0 neighbour and gives the entry no layer-0 reverse edge →
two layer-0 islands seed and never merge (every later level-0 point greedily attaches to one island only). Orthogonal to the
g17 `l_n` reverse-edge line (reproduces identically on the reverse-edge-unpatched twin).

### Reproducibility (the enabling change)
`vendor_g17/hnsw_rs_exp/src/hnsw.rs` got two experiment hooks (thread-local, default-off, zero effect when unused):
`set_level_seed(u64)` (seeds the OS-random `LayerGenerator` so a build replays) and `set_forced_levels(Vec<usize>)` (forces the
first points' levels — the causal knob). Plus `FIX_MODE: AtomicU8` selecting the search_layer empty-check behaviour (0 = current,
1 = `layer!=0 && empty`, 2 = suffix-empty). Only `hnsw_rs_exp` carries these; `hnsw_rs_patched` is untouched.

### Causal proof (`h15_force`, n2000 d32 iid, 20 builds/cell unless noted)
- forced level prefix `[1,1]` → 20/20 catastrophic (mean 25.5 % lost); `[1,0]` 0/20, `[0,1]` 0/20, `[0,0]` 0/20 (all "small"
  ~1 % loss, the g17 mode); `[2,2]`,`[2,1]`,`[3,3]`,`[2,1,2]`,`[5,5,5]`, ten-1s all → catastrophic. So the trigger is exactly
  "first two inserted points both level>=1", any upper levels.
- reverse-edge-UNpatched twin (`h15u_force`): `[1,1]` 20/20 catastrophic, `[0,0]`/`[1,0]` 0 → confirms orthogonal to g17.
- with FIX_MODE 1: `[1,1]` → 0/20 catastrophic (and `[1,1,1,1,1,1,1,1,1,1]`, `[3,3]`, `[2,1,2]`, `[5,5,5]` all 0).

### Rate vs N / dim / M / ef_construction / data (`h15_grid`, 3000 builds/cell, OS-random levels, seeded-replayable)
The predicate "first-two-both-upper" == "layer-0 graph has >1 weak component" == "catastrophic (>5 % self-miss)" in EVERY cell
(flag==comps>1==cat; flag_but_not_cat<=1, loss_without_flag=0). The rate is **1/M^2**, independent of N, dim, ef_construction, data:

| cell (n500 d32 M16 efc200 iid unless noted) | builds | catastrophic | rate (Wilson 95 %) | 1/M^2 |
|---|---|---|---|---|
| base | 3000 | 12 | 0.40 % (0.23-0.70) | 0.39 % |
| n=200 | 3000 | 8 | 0.27 % (0.14-0.53) | 0.39 % |
| dim=8 | 3000 | 7 | 0.23 % (0.11-0.48) | 0.39 % |
| dim=128 | 3000 | 7 | 0.23 % (0.11-0.48) | 0.39 % |
| efc=100 | 3000 | 11 | 0.37 % (0.21-0.66) | 0.39 % |
| efc=400 | 3000 | 10 | 0.33 % (0.18-0.61) | 0.39 % |
| **M=8** | 3000 | 47 | **1.57 % (1.18-2.08)** | **1.56 %** |
| **M=32** | 3000 | 3 | **0.10 % (0.03-0.29)** | **0.098 %** |
| clustered | 3000 | 6 | 0.20 % (0.09-0.44) | 0.39 % |
| low-intrinsic-dim 3 | 3000 | 15 | 0.50 % (0.30-0.82) | 0.39 % |

Pooled pure-iid M16 (18000 builds): 55 catastrophic = **0.306 % (0.235-0.397 %)** — consistent with H1's 0.55 % [0.33-0.92].
The M=8/M=16/M=32 trend nails 1/M^2 (and explains why H1's unit test, M=8 = 12.5 % upper, saw it). It is RARE at the production
M=16 and would be ~1/M^2 at any M.

DUPLICATES are a RED HERRING for this mode: `dup5` (5 % exact dups) shows 43.6 % of builds with >5 % **strict-id** self-miss, but
comps>1 stays at baseline (11/3000) — the graph is NOT disconnected; a point simply is not its own strict-id top-1 because its
duplicate ties at distance 0 (it IS findable at distance 0). `HnswIndex::audit` already tolerates `distance<=1e-6`, so this does
not reach memory. `dup30` → 100 % by the strict metric, comps still baseline. (Takeaway: strict-id self-miss over-counts on
duplicate-heavy data; the layer-0 connectivity check and distance-tolerant audit are the correct detectors.)

### The fix (one line; ADDITIVE to the g17 reverse-edge patch) — `results/h15_proposed_patch.md`, diff `results/h15_hnsw_layer0.diff`
```diff
-        if self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
+        if layer != 0 && self.layer_indexed_points.points_by_layer.read()[layer as usize].is_empty() {
```
VALIDATION (`h15_cmp`, experiment crate = g17 reverse-edge patch + this guard):
- rate 0: all 10 naturally-found rare seeds (100062/100870/100997/101119/101350/101403/200398/200405/200730/200900) lose
  242-653 of 2000 unpatched (FIX_MODE 0) and **0** with FIX_MODE 1 (and FIX_MODE 2). `h15_wk fix1` arm: 28/5000 flagged builds,
  **0 failures** (the `none` arm fails all 28/28).
- equivalence on the 99.6 % non-triggering builds: FNV graph hash over all layers' neighbour lists is byte-IDENTICAL between
  FIX_MODE 0 and 1 on every non-flag build (6000-build n500 identity run; the hash differs ONLY on the ~0.4 % flagged builds).
  recall@10 and build time unchanged: n5000 d32 non-flag means fix0/fix1 r@10 ef50/100/200 = .9849/.9996/1.0000 vs
  .9846/.9996/1.0000, build 2342 vs 2345 ms, 403 vs 410 us/q (noise). FIX_MODE 2 also 0-failure but changes the graph on a few
  non-catastrophic builds (upper empty layers), so the minimal `layer != 0` guard is preferred.

### Workaround with NO crate change (if the patch is not applied) — cost/benefit, `h15_wk`, n500, 5000 builds/arm
| arm | flagged | FAILS after | extra builds/build (max) | mean ms | note |
|---|---|---|---|---|---|
| none | 28 | **28** | 0 | 43.9 | status quo of the vendored crate |
| flat02 = `modify_level_scale(0.2)` | **0** | 0 | 0 | 42.9 | collapses hierarchy so ~no upper points → trigger ~never; also kills the g17 mode (g17 already recommends it). Prints a library warning line. |
| restart2 (check `get_layer_nb_point(0)==0` after 2 inserts, rebuild if 0) | 28 | 0 | 0.006 (2) | 44.2 | public-API detector, no extra scan |
| conn (rebuild while layer-0 weak-components>1) | 28 | 0 | 0.006 (2) | 44.9 | exact detector |
| audit20 / audit100 (self-query s samples, rebuild if any miss) | 28 | 1 / 0 | 0.006 (2) | 45.6 / 49.5 | = the existing `HnswIndex::audit_and_repair`; s=20 missed 1 |

So a rebuild (fresh levels) repairs it with prob ~1-1/M^2; only ~0.6 % of builds ever need the one extra build. The existing
`src/memory.rs HnswIndex::audit_and_repair(sample, max_lost, max_rebuilds)` already implements the audit arm.

### Detector cost/benefit (`h15_audit`; forced positives [1,1] vs negatives [0,0])
- EXACT layer-0 weak-component check: pos comps mean ~2.0, neg exactly 1.0, **0 false alarms, 0 misses** in every grid (18k+
  builds). Cost ~0.5 ms (n500) / 3.9 ms (n2000) / 11 ms (n5000) — a fraction of build time. This is a PERFECT and cheap detector,
  strictly better than sampled self-query for this mode.
- sampled self-query audit (distance-tolerant, the memory.rs rule), detect-any-miss on positives / false-alarm on negatives:
  n2000 d32 — s=20 94 %/0, s=50 98 %/0, s=100 100 %/0; n5000 d32 — s=20 100 %/0; n2000 d128 CLUSTERED — weaker (s=100 92.5 %,
  needs s=500 for 100 %) and 1/40 false alarms at s>=50 (the normal ~1 % loss on near-duplicate cluster points). So for this mode
  prefer the connectivity check; keep the sampled audit only as the no-structural-access fallback and keep its `distance<=1e-6`
  tolerance (else duplicates false-alarm).

RECOMMENDATION: apply the one-line `layer != 0` guard to `vendor_g17/hnsw_rs_patched/src/hnsw.rs` (director's call — R7-A may not
edit that dir). Zero cost, keeps the g17 fix, removes the mode entirely. If not applied: `modify_level_scale(0.2)` at
`HnswIndex::new` is the cheapest no-crate-change mitigation (removes BOTH HNSW modes, as g17 found), else keep
`audit_and_repair` (one connectivity check or ~100-sample distance-tolerant audit per build + ~0.6 % rebuilds).

### NEGATIVES / LIMITS
1. All data synthetic; the mechanism is data-independent (depends only on the random first-two levels), confirmed on iid, clustered,
   low-dim, and duplicate data. 2. Rate measured at N<=20000, M 8/16/32, efc 100-400, d 8-384; the 1/M^2 law held in every cell,
   but I did not test N>20k or M<8. 3. The forced-level hook changes `LayerGenerator`; I verified the default path (no hook set)
   is byte-identical graph to the stock crate on 752 non-flag builds, so the hook does not perturb normal builds. 4. Equivalence of
   the fix is graph-hash identity on non-flag builds (6000 n500) + unchanged recall/build on n5000 d32 / n2000 d384 / n5000 d128
   clust; I did not run >=5000 builds at every (N,dim) — the fix only alters behaviour on the ~0.4 % flagged builds, which is why
   identity suffices. 5. `h15_wk` failure counts are on 5000 builds/arm at n500; extrapolate the 0-failure claim with the Wilson CI.
   6. Loaded box: ms/us indicative.

### FILES (nothing in repo Cargo.toml / src / registry touched)
- `vendor_g17/hnsw_rs_exp/` (2nd vendored copy = g17 reverse-edge patch + experiment hooks `set_level_seed`/`set_forced_levels`/
  `FIX_MODE`; read first: diff vs `hnsw_rs_patched` is only those hooks + the `layer!=0` guard behind FIX_MODE), `hnsw_rs_exp_unp/`
  (reverse-edge-unpatched twin).
- `vendor_g17/harness2/` (standalone `[workspace]` cargo project, `--offline`, deps from the on-disk hnsw_rs 0.3.4 cache): bins
  `h15_find` (reproducible catastrophe search), `h15_trace`/`h15_anal` (structure dump), `h15_force` (causal forced-level test),
  `h15_grid` (rate grid CSV), `h15_cmp` (fix identity + recall/latency), `h15_wk` (no-crate workarounds), `h15_audit` (detector
  cost/benefit). `vendor_g17/harness2u/` = same bins on the unpatched twin. `vendor_g17/harness2/run_grid.sh`.
- `results/h15_proposed_patch.md`, `h15_hnsw_layer0.diff`, `h15_grid_*.csv`, `h15_cmp_ident_n500.csv`, `h15_cmp_perf_*.csv`,
  `h15_cmp_naturalseeds.txt`, `h15_wk_n500_{A,B}.csv`, `h15_audit_*.txt`, `h15_lossfrac_forced11.csv`, `h15_force_*.txt`,
  `h15_unpatched_forced.txt`, `h15_find_s*.log`, `h15_trace_100062.txt`, `h15_anal_100062.txt`, this file, `lessons_h15.md`.

### RERUN
```
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r1 RAYON_NUM_THREADS=2
cd /c/Users/mariu/neural-mesh/vendor_g17/harness2 && cargo build --release --offline        # builds all h15_* bins
T=$CARGO_TARGET_DIR/release
$T/h15_force.exe --prefix 1,1 --builds 20 ; $T/h15_force.exe --prefix 0,0 --builds 20        # causal: [1,1] all cat, [0,0] none
$T/h15_force.exe --prefix 1,1 --builds 20 --fix 1                                             # fix: 0 cat
cd /c/Users/mariu/neural-mesh
$T/h15_grid.exe --n 500 --dim 32 --m 8  --builds 3000 --seed0 5000000 | awk -F, 'NR>1&&$13>0.05*$2{c++}END{print c" cat (expect ~1.56%)"}'
$T/h15_grid.exe --n 500 --dim 32 --m 32 --builds 3000 --seed0 6000000 | awk -F, 'NR>1&&$13>0.05*$2{c++}END{print c" cat (expect ~0.10%)"}'
$T/h15_cmp.exe --n 2000 --dim 32 --builds 1 --seed0 100062                                    # fix0 559 lost, fix1/fix2 0
$T/h15_cmp.exe --n 500 --builds 6000 --seed0 40000000 > /tmp/id.csv                           # identity: fix0==fix1 hash on all non-flag
```
