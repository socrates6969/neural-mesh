# H6 / R4-B3 (MEMORY-REDUCTION track): compression of the knowledge store + retrieved context, downstream quality held?

Scope: OUR vector store and retrieved-context footprint only (not Claude's internal memory). Code: `src/bin/h6_store.rs` (closed loop), `src/bin/h6_docq.rs` (doc-level retrieval, context frontier), `src/bin/h6_gate.rs` (size gating), `src/bin/h6_common/mod.rs`.
Nothing in `src/*.rs` edited; proposals in `results/h6_proposed_patch.md`. Outputs: `results/h6_loop_{A,B}.{json,log}`, `h6_loopbase_{A,B}.*`, `h6_docq.{json,log}`, `h6_ctx.{json,log}`, `h6_gate_grid_{10000,100000,1000000}.{json,log}`, `h6_gate_rule.{md,json,log}`.
Label on every number: SYNTHETIC until validated on real traffic (g1 templated corpora with real MiniLM/BGE embeddings; g16 seeded worlds; g15 synthetic mixture). No intelligence/IQ claim: this is storage/footprint, quality is "routing success of the knowledge layer".
Build: `CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=2 cargo build --release --offline --bin h6_store --bin h6_docq --bin h6_gate`.

## Protocol (pre-declared before running)
Equivalence margin +-0.01 on H2000 task success (and on F1@3 / hit@1 for doc retrieval), TOST on paired differences vs the f32 flat store of the SAME policy. Closed loop = the g16 engine copied into `h6_common::run6` (bit-exact vs `g16::run`: `h6_store check` max|d| = 0.0). Policies: R0 (g1 default retrieval) and the FROZEN g16 FINAL[mini-hybrid] (read from results/g16_picks.json). 8 worlds (sib16, rand16, sib16-mixed50, sib16-weakid50, sib16-drift, ctl-irrel, sib64, ctl-oneoff). No tuning in this work at all: held-out corpus A seeds 31-60, confirm corpus B seeds 61-90 with the identical arm list (it is a re-run, not a tune). Holm family = the 8 worlds of each arm (declared; the max raw TOST p is also stored in the json so a reader can apply a larger family). Doc-level units are per-concept means (240 concepts) to avoid pseudo-replication across the ~9 phrasings per concept. 30 seeds are resampled worlds, not independent corpora.
Context for the margin: retrieval itself is worth H2000 +0.077/+0.082 (sib16 R0/FINAL), +0.082/+0.101 (rand16), +0.039/+0.036 (drift), +0.024/+0.027 (sib64), +0.009/+0.008 (weakid50), and costs -0.008..-0.020 in the controls (`h6_loopbase_*`). So +-0.01 is <=12 % of the gain in sib16/rand16 but 40 % in sib64 and >100 % in weakid50: equivalence in those two worlds is a weak statement; look at the CIs in the json.

---
### VERIFY REQUEST - R4-B3 - quantised knowledge store in the closed loop (TOST +-0.01, both policies, corpus A held-out + corpus B confirm)
Rerun (~90 s each + build; box load changes nothing, the loop is deterministic):
```
B=$CARGO_TARGET_DIR/release/h6_store.exe
$B check                                   # run6 == g16::run, max|d| 0
$B loop --corpus A                         # seeds 31-60 -> results/h6_loop_A.json
$B loop --confirm                          # corpus B, seeds 61-90, same arms -> results/h6_loop_B.json
$B loopbase --corpus A ; $B loopbase --confirm     # no-retrieval baseline (retrieval gain context)
```
Row bytes = dense + 320 B sparse tf-idf (g16 estimate, NOT compressed here) + 16 B meta; f32 flat = 1872 B/row. "equiv" = all 8 worlds TOST-equivalent after Holm/8. Store at loop scale is only 2000 rows = 3.7 MB, so absolute savings are tiny; the table is per-row/percent and scales linearly with the store.

| store arm (hybrid row) | B/row (saving) | rows@2000 | R0 A / R0 B equiv | FINAL A / FINAL B equiv | worst-world H2000 diff (R0A, R0B, FA, FB) | retrieval gain kept (FINAL, 4 gain worlds, A/B) |
|---|---|---|---|---|---|---|
| flat f32 (incumbent) | 1872 | 2000 | - | - | 0 | 100/100 % |
| flat f16 | 1104 (-41 %) | 2000 | yes / yes | yes / yes | 0, -.0006, 0, -.0013 | 100/100 % |
| **flat block-fp16** | **1152 (-38 %)** | 2000 | yes / yes | yes / yes | **0.0000 in all four (identical decisions)** | 100/100 % |
| **flat int8** (f32 query) | **724 (-61 %)** | 2000 | yes / yes | yes / yes | -.0004, -.0004, -.0022, -.0009 | 99/100 % |
| flat int8 (int8 query) | 724 | 2000 | yes / yes | yes / yes | -.0004, -.0003, -.0016, -.0008 | 99/99 % |
| flat turbovec 4-bit, no rerank | 532 (-72 %) | 2000 | yes / yes | yes / **NO** | -.0015, -.0019, -.0025, -.0045 | 98/98 % |
| flat turbovec 3-bit, no rerank | 484 | 2000 | yes / yes | yes / **NO** | -.0014, -.0014, -.0059, -.0045 | 100/99 % |
| flat turbovec 2-bit, no rerank | 436 (-77 %) | 2000 | yes / yes | **NO / NO** | -.0050, -.0021, -.0061, -.0018 | 102/101 % |
| turbovec 4-bit + f32 rerank 4x (f32 copy NOT in RAM) | 532 RAM + 1536 off-RAM | 2000 | yes / yes | yes / yes | 0, -.0010, 0, -.0001 | 100/100 % |
| turbovec 2-bit + f32 rerank 4x / 16x | 436 RAM + 1536 off-RAM | 2000 | yes / yes | yes / yes | <= -.0010 | 100/100 % |
| exact-duplicate suppression (f32 / block-fp16 / int8 rows) | 1872 / 1152 / 724 | **209 (-90 %)** | yes / yes | yes / **NO** (B sib16 -.0085, CI [-.018,+.001]) | -.0067, -.0024, -.0023, -.0085 | 101/95 % |
| anchor-summary rows, thr frozen | 1872 | **111 (-94 %)** | yes / yes | **NO / NO** (B: sib16 -.0066, drift -.0078 CI [-.0136,-.0019]) | -.0019, -.0033, -.0018, -.0078 | 102/94 % |
| summary rows thr .55 / .65 (post-hoc, my choice of grid points) | 1872 | 46 / 77 | NO / NO | NO / NO | -.028..-.042 / -.009..-.011 | 73-76 % / 95-97 % |
| k=1 / 2 / 3 / 5 / 12 (context truncation of the vote) | 1872 | 2000 | yes (all) | k1,k2: yes / **NO** (B sib16 -.0044 CI [-.014,+.005]); k3,5,12 yes/yes | k5 ~0 | 98-104 % |
| adaptive k (score drop .03/.06/.10, kmax 8) | 1872 | 2000 | yes / yes | yes / **NO** (B sib16 ~-.005) | -.0011..-.0053 | 97-103 % |

How to read "NO": TOST inconclusive at +-0.01 after Holm/8, NOT proven harm. Almost all NO cells have a CI that straddles 0 and often a positive upper bound (e.g. FINAL turbovec-2bit corpus A: sib16 +.0032 [-.0037,+.0101]); a perturbed similarity matrix re-randomises the trajectory (chaotic), so the paired-diff variance is large relative to a +-0.01 margin. The few negative-leaning NO cells: summary rows (sib16/drift in B), duplicate suppression with FINAL (sib16 in B).
Mechanism of the "no loss": the loop only uses a hybrid similarity vs a threshold (.75) and top-k neighbour votes; quantisation noise (<=0.005 in cosine for 4-bit) rarely flips a neighbour across the threshold, and the neighbour set diverges only in near-ties.

### Bytes saved per quality held (closed loop, TOST-equivalent in ALL four policy x corpus cells)
block-fp16: -38 % row (-47 % dense) with ZERO measured change; int8: -61 % row (-75 % dense) at <= -0.0022 worst-world H2000; f16: -41 %. Anything below int8 without an exact rerank was NOT equivalent in all cells. The 320 B sparse part limits the hybrid-row saving (Amdahl): compressing it is the next lever and was not tried.

---
### VERIFY REQUEST - R4-B3 - document/chunk-level retrieval with quantised stores (recall@k vs f32, F1@3, hit@1, MRR, TOST)
Rerun (~1 min): `$CARGO_TARGET_DIR/release/h6_docq.exe docq` -> `results/h6_docq.{json,log}`. 2 corpora x {MiniLM+spell, BGE+instruction} x {780 docs, ~3k chunks}; 8 sets; unit = per-concept mean; TOST +-0.01 Holm over (variant x {recall@10, F1@3, hit@1}) per set.
| variant | B/vec mini / bge | recall@1 (worst set) | recall@10 range | F1@3 diff range | hit@1 diff range | F1 TOST equiv | hit@1 TOST equiv |
|---|---|---|---|---|---|---|---|
| f16 | 768 / 1536 | 0.999 | 1.000 | 0..+.0002 | -.0005..+.0005 | 8/8 | 8/8 |
| **block-fp16** | 816 / 1632 | 1.000 | 1.000 | **0** | **0** | 8/8 | 8/8 |
| int8 / int8(query int8) | 388 / 772 | 0.993 / 0.991 | 0.996-0.999 / 0.995-0.998 | -.0007..+.0002 | -.0010..+.0010 | 8/8 | 8/8 |
| turbovec 4-bit | 196 / 388 | 0.968 | 0.981-0.985 | -.0013..+.0007 | -.0042..+.0036 | 8/8 | 7/8 |
| turbovec 3-bit | 148 / 292 | 0.943 | 0.964-0.972 | -.0014..+.0010 | -.0057..-.0005 | 8/8 | 4/8 |
| turbovec 2-bit | 100 / 196 | 0.903 | 0.938-0.950 | -.0106..-.0004 | -.0130..-.0005 | 5/8 | 1/8 |
| tv 4-bit / 2-bit + f32 rerank 4x, 2-bit rerank 16x | 196 / 100 RAM (+f32 elsewhere) | 1.000 | 0.999-1.000 | 0 | 0 | 8/8 | 8/8 |
Retrieval recall@10 (id-set overlap with f32) is NOT "held" for turbovec without rerank (0.94-0.985: fails a +-0.01 non-inferiority bar in every set) even where the end task metric is; with an exact rerank it is 0.999-1.000. These stores are tiny (<= 3k vectors), so this table demonstrates quality, not byte savings in absolute terms.

---
### VERIFY REQUEST - R4-B3 - retrieved-context compression (k, adaptive k, byte budget, exact-dup suppression): quality vs injected chars
Rerun (~10 s): `$CARGO_TARGET_DIR/release/h6_docq.exe ctx` -> `results/h6_ctx.{json,log}`. MiniLM+spell dense retrieval, relevant = the 2 docs of the concept (BY CONSTRUCTION the F1 knee is at k=2), 30 % of docs re-ingested as exact duplicates (labels I attach).
Corpus A (B in json, same shape): fixed k=1 / 2 / 3 / 4 / 6: chars 328 / 654 / 981 / 1308 / 1963, F1 .614 / **.684** / .671 / .606 / .497, hit_any .920 / .944 / .967 / .974 / .985. Adaptive score-drop (kmax 6) .05: 597 chars F1 .655; .10: 827 chars F1 .670 (both below fixed k=2). Byte budget 900 chars == k=2 (docs are ~327 chars each, so a byte budget adds nothing over k here; it differs only for variable-length docs: not tested). Budget < 654 chars degenerates to k=1.
**Knee = k=2 / ~650-900 chars**: -33 % chars vs the common default k=3 AND +0.013 F1 (A) / +0.021 (B); F1 drops steeply past k=3 (precision), hit_any gains only +0.02-0.03 from k=3 to 6 at 2-3x the bytes.
Exact-duplicate suppression (cosine>=.999 equivalent: identical vector): F1 at k=2 .684 -> .7195 (A, +.0355) and .687 -> .712 (B, +.025); rows 1014 -> 780 (-23 %); context chars unchanged (it frees slots, it does not shrink the context). Consistent with g16 E2 (+.021 at k=3). Stale-revision detection by cosine was NOT retried (g16: failed).
Closed-loop analogue (table above): shrinking k to 3-5 in the VOTE is TOST-equivalent but saves no store bytes (it only reads fewer neighbours); k<=2 and adaptive k are inconclusive/slightly negative for FINAL on corpus B.

---
### VERIFY REQUEST - R4-B3 - size gating: own scans at N = 1e4, 1e5, 1e6 (synthetic d=384), RAM <= 3 GB, f32 copy in RAM vs on DISK (unbuffered QD1)
Rerun (RAM check first; box free RAM was 3.5-4.5 GB; 1e6 run peaked at **2,254 MB** working set, aborts below `--min-free-gb`):
```
G=$CARGO_TARGET_DIR/release/h6_gate.exe
$G grid --n 10000 --disk-cold ; $G grid --n 100000 --disk-cold ; $G grid --n 1000000 --nq 100 --reps 2 --disk-cold --min-free-gb 3.2
$G rule                                  # fits the cost model, prints decision table + validation
```
Single thread, p50 us (mean of reps), recall@10 vs exact f32; box load: calib probe 2.1 ms normal, 3.1 ms during the 1e6 run => 1e6 latencies are inflated ~1.4x (flat f32 67 ms here vs 45 ms in g15's quiet run).
| config (RAM B/vec; +f32 copy) | N=1e4 p50 / recall | N=1e5 p50 / recall | N=1e6 p50 / recall |
|---|---|---|---|
| flat f32 AVX2 (1536) | 470 / 1.000 | 5151 / 1.000 | 67434 / 1.000 |
| flat block-fp16 (816) | 311 / 1.000 | 3733 / 1.000 | 44012 / 1.000 |
| flat int8 (388) | 136 / 0.997 | 1672 / 0.974 | 25351 / 0.961 |
| int8 + f32 rerank 4x, f32 in RAM (388 + 1536) | 150 / 1.000 | 1602 / 1.000 | 18720 / 1.000 |
| int8 + f32 rerank 4x, f32 ON DISK (388) | 4465 / 1.000 | 6228 / 1.000 | 26113 / 1.000 |
| turbovec 4-bit, no rerank (196) | 198 / 0.965 | 1192 / 0.848 | 15103 / 0.774 |
| turbovec 4-bit + rerank 4x, f32 in RAM (196 + 1536) | 244 / 1.000 | 1185 / 1.000 | 14290 / 1.000 |
| turbovec 4-bit + rerank 4x, f32 ON DISK (196) | 4700 / 1.000 | 5798 / 1.000 | 29021 / 1.000 |
| turbovec 2-bit + rerank 16x, f32 in RAM / on disk (100) | 280 / 1.000 ; 17445 | 1304 / 1.000 ; 20975 | 10519 / 0.972 ; 36621 |
| turbovec 2-bit + rerank 4x (100) | 127 / 0.998 | 1044 / 0.957 | 8920 / **0.742** |
| own RaBitQ-style 1-bit (68; randomised-Hadamard rotation, LUT scan) | 415 / 0.852 | 4239 / 0.399 | 70785 / 0.182 |
| RaBitQ-style 1 bit + rerank 32x (f32 in RAM) | 614 / 1.000 | 4597 / 1.000 | 46884 / 0.927 |
Disk rerank is a measured cost, not a free lunch: 40 unbuffered QD1 random 1536 B reads = 4.5-4.8 ms/query (~115 us each, `FILE_FLAG_NO_BUFFERING`, i.e. cold-SSD, not page cache); 160 reads (16x) 18 ms; at 1e6 the 4x disk rerank costs +9 ms on this loaded box for int8 (reads queue behind the scan). QD>1 / mmap prefetch NOT measured.
Gating rule (`h6_proposed_patch.md` P2): choose the LEAST compressed backend with RAM <= budget, modelled p50 <= latency budget, recall >= 0.99, from a linear cost model fitted on the 3 grid sizes. Validation against the measured rows at the 3 sizes x 3 latency budgets (1/5/20 ms) x 3 RAM budgets (128 MB/512 MB/2 GB): **27/27 cells agree** with the measured-least-compressed-feasible backend (note: ~20/27 of those cells are trivially f32 or NONE; the interpolated 3e5 and extrapolated 3e6/1e7 rows are model output, not measurements). Selected rows (p50 budget; RAM budget): N<=1e4: flat f32 everywhere; 3e4, 1 ms: block-fp16; 1e5, 5 ms: block-fp16; 1e5, 20 ms: f32 (512 MB+); 3e5, 5 ms, 2 GB: tv4+rerank (f32 in RAM); 1e6, 20 ms, 2 GB: int8+rerank (f32 in RAM); anything with a 1 ms budget at N>=1e5 or 5 ms at N>=1e6: NONE = needs an ANN graph (g15: HNSW ~6.2 KB/vec, p50 0.4 ms at 1e5; not re-measured here).
**Crossover (measured): flat f32 stays the right choice up to ~2e4 vectors (p50 <= ~1 ms) and, when RAM is not binding, up to ~1e5-3e5 at a 5-20 ms budget; compression matters for LATENCY before it matters for RAM (1e6 x 1536 B = 1.5 GB fits the 3 GB cap).** Below 1e5, block-fp16 is the only compression that is free in quality and it saves RAM, not time.

---
## Honest negatives / failures
1. turbovec 2-4-bit without rerank is not a safe knowledge-store scan: doc recall@10 vs f32 0.94-0.985; closed loop FINAL fails TOST in some world of both corpora (4-bit on B, 2-bit on A and B, 3-bit on B); recall at 1e6 synthetic 0.77 (4-bit).
2. The rerank rescue needs the f32 copy somewhere: in RAM it removes the saving (196 + 1536 B/vec > 1536), on cold disk it costs +4.5 ms/query at 4x. At <= 1e5 vectors flat f32/block-fp16 beat disk-rerank on both RAM-adequate budgets and latency.
3. Anchor-summary rows (-94 % rows) lose 6 % of the retrieval gain on corpus B (FINAL) with the frozen threshold; changing the threshold (.55/.65) is worse (the centroid-mean score has a different scale). No retuned variant was found, so no recommendation.
4. Exact-duplicate suppression in the VOTE loop (-90 % rows) is neutral for R0 but inconclusive/slightly negative (-.0085, CI crosses 0) for FINAL sib16 on B. Its doc-level benefit (F1 +.025-.036) is real in the duplicate-injected store.
5. Adaptive k and byte budgets do not save store bytes in the routing vote and are no better than a tuned fixed k at the doc level (g16 E1 conclusion confirmed here).
6. My RaBitQ-style 1-bit scan (own code, LUT scan not SIMD fast-scan, randomised Hadamard instead of a dense random rotation, no per-query bit-plane quantisation) is slower than f32 flat at 1e5-1e6 and recalls 0.18-0.40 raw: a negative implementation result, not a verdict on RaBitQ. turbovec (same quantiser family) is the multi-bit option measured.
7. The sparse tf-idf part (320 B) of the hybrid row was left uncompressed; with it, int8 saves 61 % of a row not 75 %.
8. Equivalence statements are only as strong as the margin: +-0.01 is 40 % of the sib64 gain and larger than the weakid50 gain.
9. The gating rule is fitted on 3 N and on a loaded box (p50 ~1.4x inflated at 1e6); extrapolations (>1e6) and the 3e5 row are model output; HNSW and IVF were not part of the grid.
10. Corpora are templated synthetic text (near-duplicate heavy); synthetic mixture at 1e6 is near-equidistant, which makes approximate-recall numbers at large N pessimistic (not verified on a real >=1e5 corpus: none exists in results/).

## Recommended store configuration (opt-in, defaults unchanged)
- Dense vectors: **block-fp16 as the free tier** (identical decisions in the loop, 0 doc-level loss, -38 % row); **int8 when you need -61 % row** (TOST-equivalent in 4/4 loop cells, doc recall@10 >= 0.995, worst H2000 -0.0022; below N~1e5 only; above, add an exact rerank because id-set recall@10 drops to 0.961 at 1e6).
- Do not use turbovec / RaBitQ-style codes without an exact rerank; use them with rerank only when RAM binds (f32 on disk) AND a ~5 ms disk-rerank latency is acceptable, or when the f32 copy fits in RAM and the latency budget (not RAM) is what forces compression (tv4 + rerank 4x: 14 ms vs 67 ms flat f32 at 1e6, same recall).
- Gate by size with the rule above: f32 flat <= ~2e4 rows; block-fp16 up to ~1e5; beyond that pick by latency budget (int8/tv4 + rerank with f32 in RAM) and only then by RAM (f32 on disk); an ANN graph when a <= 1 ms budget applies at >= 1e5.
- Context: k=2-3 documents (~650-1000 chars) is the knee in this synthetic setup (k=2 by construction); suppress exact-duplicate documents before retrieval (+0.025-0.036 F1 when duplicates exist); do not use summary rows or adaptive-k for byte savings.
- All of this must be re-measured on real traffic before any default changes; nothing here is applied.
