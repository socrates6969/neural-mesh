### VERIFY REQUEST — R9-A — H17: how far can the hash-embedding (synthetic encoder) results be trusted? Real MiniLM vs HashEmbedding on the SAME corpora, retrieval + closed-loop routing + CompactRouter TOST + cost

LABELS (read first). REAL = the saved MiniLM-L6 vectors (precomputed offline by the existing cognee-venv onnxruntime; results/h10_emb_*.f32, g1_emb_*_mini.f32), the real doc corpora (h10 A 6831 paragraphs, B 4181), real query text. SYNTHETIC = QWERTY typos, the g1 templated corpora, every routing outcome (competence tables, context effect, Bernoulli draws). Nothing here is real traffic; "validated on real traffic" is NOT met. No crate/model/download; no ort/fastembed (not built, not audited); src/ and Cargo.toml untouched. The Rust crate has no encoder: "real" = precomputed vectors; "hash" = `neural_mesh::embedding::HashEmbedding` / `knowledge::hash_embed` computed live.

Three encoders are compared, because "hash" is ambiguous in this repo:
* **mini** = real MiniLM (dim 384).
* **hash** = `HashEmbedding::new(384).embed(text)` AS SHIPPED: whitespace split, case-sensitive, punctuation kept, signed feature hashing, no idf. (Repo default dim is 64; not tested, it would only be worse.)
* **hashtok** = same hashing over lowercase [a-z0-9]+ tokens (a fairer bag-of-words). This is also what g1's own "hash" model does (`words()`), so the h17_g1 A-D numbers are hashtok-type.

## BLUNT SUMMARY

1. **The hash encoder is NOT "BM25-like". It is far worse than BM25** and the hybrid built on it is just BM25. Real docs, mix25 MRR (A held): MiniLM dense+spell .292, BM25+spell .284, **hash dense+spell .064**, hashtok .123. g1 A held: MiniLM .950, BM25 .901, **hash .591**. Cause: no idf/stopword weighting and 384-d collisions, so the cosine is dominated by common words. Tuned hybrid alpha (weight on dense) is .7 for MiniLM but .3 (real) / .1 (g1, grid edge) for hash: the hash hybrid is BM25 (hybrid - BM25 MRR within +-.002 on real docs (n.s.) and on g1 A-D (|d| <= .002, two of five splits Holm-significant but negligible)).
2. **Several headline retrieval conclusions FLIP, others only change size** (list in section 4). Flips: dense vs BM25 (tie/dense-better with MiniLM; BM25 wins by .12-.31 MRR with hash), "hybrid adds value over lexical alone" (+.048 MiniLM, 0.000 hash), g1 hybrid F1@3 (-.07 MiniLM, +.12 hash), adaptive-k vs fixed-k and the exact-duplicate-suppression gain at k=2 (both reverse/vanish under hash), the end-to-end retrieval gain on topic-dependent worlds (+.022* MiniLM, -.002 n.s. hash), and "hybrid ties dense end-to-end" (ties with MiniLM, small Holm wins with hash because the hash dense baseline is so weak).
3. **What survives the move to a real encoder is the router-layer engineering, not the retrieval science:** CompactRouter (u16/q16) vs the f64 recommended router is TOST-equivalent (+-0.005) in all 8 worlds for every encoder's anchors (54 anchor-arm x encoder x split cells, 432 world tests; worst |diff| .0017, widest 95%-CI end .0034); CompactRouter<f64> is bit-identical. Spell correction helps typo queries under every encoder (sign preserved) but the size is 1/5 (hash on real docs: +.005 n.s. in A) to 2x (hash on g1) of the MiniLM size. Control-world "wins" that are front-learning-speed effects (Tctx, ctl-irrel +.02..+.07) are encoder-independent.
4. **Trust rule:** hash-embedding results may be used for plumbing, determinism, byte/latency and router-state equivalence tests. They must NOT be used to rank retrieval levers (dense vs BM25 vs hybrid, alpha, thresholds, k, dedup), to size retrieval gains, or to set cosine-scale thresholds. Any retrieval claim made on a hash world must be re-run on real vectors before it is believed. All g1 retrieval/routing numbers in earlier rounds that used `hash` arms should be read as BM25-weaker-than-BM25 stand-ins.
5. **Cost of the real path** (section 5): MiniLM = 90.4 MB model + 0.47 MB tokenizer, +103 MB RSS at load (peak 359 MB), 2.4 ms per query text (p95 4.1), 4.6 ms per doc paragraph, 1536 B per vector; hash = 0 B model, 0.67 us per query / 1.39 us per paragraph (about 3600x faster), 1536 B (dim 384) or 256 B (dim 64). Building the h10 world-A index with MiniLM took 189 s (49,214 texts). The dense scan itself costs the same for both (same dim).

## 1. Protocol (as g1/h10/h13; nothing tuned on evaluation seeds)

* Retrieval, real docs: copy of the h13 harness (src/bin/h17_retrieval.rs + h17_common) with an embedder switch `H17_EMB=mini|hash|hashtok`. Tune = A seeds 1-8 (only z-fusion weight, per encoder and flavour, grid extended to .1..1.0 because hash optimum sat at the h13 grid edge .4), frozen on A held 31-60, A confirm 61-90, B confirm 61-90 (independent text). Holm over 126 tests per split; "robust" = Holm-significant under BOTH the seed t-test and the query-cluster bootstrap (4000 reps).
* g1 A-D: src/bin/h17_g1.rs (g1_retrieval engine, models hash+mini; alpha tuned on corpus-A tune concepts (concept%5==0) and frozen; held-out concepts only; each seed = 300 held-out queries; A 31-60, A 61-90, B/C/D 61-90; paired t-CI over seeds, Holm per split over 32 tests); plus the h13 g1 bridge for A,B per encoder (hybrid/RRF/rerank).
* Context frontier (fixed k / adaptive drop / byte budget, exact-duplicate dedup): the h6 `ctx` experiment on A-D for hash and mini (30% exact duplicate re-ingestion; per-concept units; TOST +-.01 vs best-F1 policy).
* Closed loop: copy of h13_route (src/bin/h17_route.rs), the g16 knowledge layer, 8 worlds, tune A 1-8 (admission floor grid extended to {0,.05,...,.7}, min_n, vote k), held A 31-60, frozen A 61-90 and B 61-90; baselines = the 6 pre-declared + global-ts; Holm per family; TOST +-.01 for controls. The topic clusters, keyword features and competence tables are the SAME h13_world fixtures for every encoder (MiniLM k-means clusters), only the retrieval top-lists change, so differences are the encoder's effect on retrieval -> anchor -> routing. A hash-defined topic structure would be a different world (not run).
* CompactRouter TOST: same loop, front = `defaults::recommended_router` (f64) vs `CompactRouterG<Fx16,Q16>` (default config) vs `CompactRouterG<f64,f64>`; anchor feature `kn:<cluster of the retrieved top-1>` (+ in 2 arms an open-vocabulary `kd:<top-1 doc id>`), arms = safe/g1 gates x {dense_s, bm25_s, hybz_s} with the tuned picks (not re-tuned per front), pooled A seeds 31-90 (n=60) and B 61-90 (n=30), H2000 paired by seed, margin +-0.005 pre-declared (as h5), Holm over 8 worlds, CERTIFIED95 = Holm-TOST AND 95% CI inside the margin.
* Reproduction check: with `H17_EMB=mini H17_THRGRID=orig` and the original grids the harness reproduces results/h13_retrieval.json (1620 level values) and h13_route.json (4248 level values) with 0 differences. Disclosed changes vs h13: (a) z-grid .1..1.0 and admission-floor grid extended (all encoders, identical for all; mini picks therefore differ slightly from h13: e.g. safe|dense_s floor .4->0.0; mini conclusions unchanged); (b) h13's RSF alpha tuning only searched the first 2 grid points (`if base == 4 { 2 }`), fixed here (RSF is not a headline, mini RSF pick still .5); (c) rerank first-stage alpha stays .7 for every encoder (not tuned for hash; rerank is not reported); (d) h17_route adds the compact-router section and `doc_feat`; nothing else changed.

## 2. Doc-level retrieval on REAL text (mix25 = 75% clean + 25% typo queries; hit@1 / MRR / F1@3)

| split | system | MiniLM | hash (shipped) | hashtok |
|---|---|---|---|---|
| A held | dense raw | .190/.277/.148 | .044/.061/.029 | .074/.114/.056 |
| A held | dense+spell | .201/.292/.156 | .041/.064/.033 | .079/.123/.061 |
| A held | BM25+spell | .195/.284/.156 | same | same |
| A held | hybrid z+spell | .228/.332/.183 | .195/.284/.156 | .193/.283/.156 |
| A confirm | dense+spell | .193/.284/.151 | .038/.064/.033 | .076/.121/.059 |
| A confirm | BM25+spell | .196/.284/.155 | same | same |
| A confirm | hybrid z+spell | .229/.332/.183 | .201/.285/.153 | .193/.282/.154 |
| B confirm | dense+spell | .164/.244/.130 | .031/.054/.027 | .075/.114/.060 |
| B confirm | BM25+spell | .149/.222/.121 | same | same |
| B confirm | hybrid z+spell | .184/.273/.146 | .148/.224/.121 | .146/.222/.120 |

P@3 / R@3 for every cell and system are in results/h17_tables.md (T1), e.g. A held dense+spell P@3/R@3 .121/.266 (MiniLM) vs .026/.054 (hash) vs .047/.102 (hashtok); hybrid .140/.314 vs .120/.265 vs .120/.266. Delta hash - MiniLM (MRR, dense+spell): -.228 (A held), -.220 (A conf), -.190 (B conf); hybrid z+spell: -.048 / -.047 / -.049 (the hash hybrid is exactly BM25). The hash rows show the shipped encoder's ordering is BM25 >> dense.

Contrasts, MRR mix25, diff [cluster-bootstrap 95% CI], + = Holm-robust (A held / A conf / B conf):
* hybrid z+spell - dense+spell: MiniLM +.046 [+.033,+.059]+ / +.045 [+.032,+.058]+ / +.027 [+.013,+.042]+; hash +.224 / +.224 / +.184 (all +); hashtok +.162 / +.161 / +.114 (+).
* BM25+spell - dense+spell: MiniLM -.003 [-.020,+.015] / -.004 / -.022 [-.041,-.003] (all n.s.: a tie); hash +.224 [+.206,+.242]+ / +.223 / +.183 (+); hashtok +.164 / +.163 / +.115 (+).
* hybrid z+spell - BM25+spell: MiniLM +.048 [+.039,+.059]+ / +.049 / +.049 [+.037,+.062]+; hash +.000 [-.004,+.004] / +.001 / +.002 (n.s.); hashtok -.002 / -.001 / -.002 (n.s.).
* spell gain, dense (spell - raw), all-queries MRR: MiniLM +.015+/+.015+/+.019+; hash +.005 [-.003,+.012] n.s. / +.005 n.s. / +.024+; hashtok +.010+/+.011+/+.015+. On the typo slice: MiniLM +.065/+.063/+.066, hash +.012/+.014/+.029, hashtok +.039/+.040/+.052. Clean slice cost: MiniLM -.002/-.002/-.001, hash 0/+.001/+.018.
* spell gain on the hybrid (hybz_s - hybz_raw): +.018/+.017/+.018 (MiniLM), +.018/+.017/+.026 (hash), +.018/+.018/+.019 (hashtok): identical across encoders because it is carried by BM25.

## 3. g1 templated corpora A-D (SYNTHETIC; g1 engine; hash here = lowercase-word hashing; held-out concepts)

hit@1 / MRR / F1@3, frozen alphas (MiniLM raw .7 spell .8; hash .1/.1 = grid edge):

| corpus split | BM25+spell | hash dense+spell | hash hybrid+spell | MiniLM dense+spell | MiniLM hybrid+spell |
|---|---|---|---|---|---|
| A 31-60 | .858/.901/.402 | .451/.591/.286 | .857/.901/.404 | .920/.950/.628 | .944/.962/.554 |
| A 61-90 | .852/.897/.399 | .460/.595/.286 | .851/.897/.400 | .916/.946/.619 | .941/.959/.546 |
| B 61-90 | .832/.880/.391 | .456/.593/.285 | .835/.882/.393 | .910/.944/.620 | .927/.955/.555 |
| C 61-90 | .846/.889/.395 | .472/.600/.279 | .842/.887/.395 | .928/.955/.620 | .946/.965/.551 |
| D 61-90 | .843/.892/.406 | .450/.582/.272 | .842/.892/.402 | .905/.939/.628 | .930/.956/.553 |

Delta hash - MiniLM (dense+spell MRR): -.359 / -.351 / -.351 / -.355 / -.357 (all Holm-significant); hybrid: -.061 / -.062 / -.073 / -.078 / -.064. Contrasts (MRR, t-CI over 30 seeds, all Holm-significant unless "ns"): spell gain dense MiniLM +.016/+.018/+.020/+.021/+.018 vs hash +.029/+.028/+.033/+.033/+.033 (typo slice +.065..+.109 vs +.132..+.153); hybrid - dense MRR MiniLM +.013/+.013/+.011/+.010/+.017 vs hash +.311/+.302/+.289/+.288/+.310; hybrid - dense **F1@3 MiniLM -.074/-.073/-.065/-.069/-.076 (a LOSS) vs hash +.118/+.114/+.108/+.116/+.130 (a gain)**; BM25 - dense MRR **MiniLM -.048/-.048/-.064/-.065/-.046 (dense wins) vs hash +.311/+.303/+.287/+.290/+.310 (BM25 wins)**. Bridge engine, A/B (all queries): the shipped whitespace `HashEmbedding.embed` has dense+spell WORSE than dense raw (MRR -.173 / -.179, Holm*) because spell-corrected text is lowercased/punctuation-free while the case-sensitive whitespace hash of the documents is not; MiniLM +.016/+.019, hashtok +.028/+.035. RRF (bridge, A/B MRR): hash .517/.535 vs its own dense_s .274/.288 (a gain over a broken dense leg); MiniLM .935/.922 vs its dense_s .946/.945 (a loss of -.011/-.023): the RRF verdict flips too.

### Context frontier and dedup (g1 A-D, spell queries, 30% exact duplicates in the store; per-concept units)
* MiniLM (reproduces results/h6_ctx.json exactly): best F1 = fixed k=2 (F1 .684 no-dedup, .720 with dedup, ~654 chars) or byte budget 900 (TOST-equivalent); adaptive drop is dominated (best adaptive .670 at 827 chars, -.014; with dedup -.050). Same on B, C, D (adaptive - best fixed: -.014..-.020 no-dedup, -.038..-.050 dedup).
* hash: best = adaptive drop **0.02** (F1 .336 at 703 chars no-dedup; .337 at 567 chars with dedup) vs best fixed k=1 .310 at 318 chars (A); adaptive - fixed +.026 / +.022 / +.020 / +.023 (A/B/C/D) no-dedup, +.028/.024/.020/.025 dedup. The adaptive threshold is in cosine units designed for MiniLM; hash's best is the smallest grid value (edge), so the hash conclusion "adaptive-k beats fixed k" is partly a scale artefact and NOT validated.
* Dedup effect (dedup - no-dedup, fixed k=2, per-concept paired t-CI): MiniLM +.036 [+.026,+.045] (A), +.025, +.027, +.033 (B,C,D), all CIs exclude 0; hash +.001 [-.005,+.007], +.001, -.004, -.000 (n.s.). At k=3/4 both encoders lose (MiniLM -.04/-.06, hash -.006..-.013).

## 4. Conclusions that FLIP (or change size) between hash and real embeddings

F = the qualitative conclusion reverses or vanishes; S = same sign, size changes materially. "robust" = Holm under seed-t and cluster bootstrap (real docs) / Holm-t (g1, routing).

| # | conclusion (MiniLM) | hash result | verdict |
|---|---|---|---|
| 1 | dense vs BM25 on real docs: TIE (-.003 [-.020,+.015] A, -.022 [-.041,-.003] n.s. B) | BM25 beats dense by +.224 / +.183 MRR (robust); hashtok +.164/+.115 | **F** |
| 2 | dense vs BM25 on g1: dense WINS (BM25 -.048..-.065 MRR, F1@3 -.22) | BM25 WINS (+.287..+.311 MRR) | **F (sign)** |
| 3 | hybrid adds value over BM25 alone (+.048, +.049, robust, 3 splits) | +.000 / +.001 / +.002 n.s.; hybrid = BM25 | **F** |
| 4 | hybrid z+spell > dense+spell on real docs (+.046/+.045/+.027 robust) | +.224/+.224/+.184 (robust) | S (x4-7; produced by a broken dense leg, not by fusion value) |
| 5 | hybrid on g1: MRR +.013 (B +.008 n.s. in the bridge) but F1@3 LOSS -.07..-.11 | MRR +.29..+.62, F1@3 GAIN +.11..+.27 | **F (F1@3 sign)** |
| 6 | tuned hybrid weight on dense alpha .7 (real) / .7-.8 (g1) | .3 (real) / .1 (g1, at the grid edge) | F (not transferable) |
| 7 | spell correction helps typo queries (+.065 typo MRR; +.015 all) | +.012 typo / +.005 all (n.s. in A) with shipped whitespace hash on real docs; NEGATIVE (-.17 MRR) with shipped hash on g1; +.13 with lowercase-word hash on g1 | S (sign preserved for hashtok; **F for shipped HashEmbedding on g1**) |
| 8 | spell gain carried by the hybrid: +.018 | +.018 / +.017 / +.026 | no flip (BM25 carries it) |
| 9 | adaptive-k is dominated by fixed k=2 / byte budget (-.014, -.05 with dedup) | adaptive drop .02 beats fixed k=1 (+.026) | **F (scale artefact)** |
| 10 | exact-duplicate suppression raises F1 at k=2 (+.025..+.036, CI excludes 0) | +.001 / -.004 n.s. | **F (vanishes)** |
| 11 | end-to-end (g1 gate, spell) retrieval beats the same front in topic-dependent worlds: T12 +.022* T12-drift +.023* T48 +.014* T12-noctx +.022* (A held; A conf +.024/.022/.011/.024*; B conf +.016/.014 (+.002 T48 n.s.)/+.016) | dense_s -.002 / .000 / -.006 / .000 (0 of 4); hashtok -.000/.002/-.003/-.000 | **F** |
| 12 | end-to-end hybrid vs dense at the same gate: tie (0 Holm wins, max .007) | hash g1|hybz_s - g1|dense_s +.013* +.010* +.008 +.009* (A held), +.011* +.012* +.008 +.008 (A conf), 0/4 in B; hashtok 3/4, 3/4, 1/4 | **F (small)** |
| 13 | injected-paragraph relevance and anchor-cluster accuracy of the vote arms (T12, A held): .20-.24 and .47-.52 | dense_s .04-.05 and .19; hybz_s .18 and .39-.41; hashtok .07-.08 / .25-.28 (hybz .20/.45) | S (anchor accuracy: hash dense 2.7x lower; lexical arms equal under every encoder) |
| 14 | ctx-only injection in the context-essential world Tqa: +.050* / +.049* / +.035* (A held/conf, B) | +.003* / +.004* / +.001 (hashtok +.010* / +.009* / +.003*); the tuned floor is .7 for every encoder (grid edge, h13's choice for ctx-only arms) and hash cosines rarely reach it (vote share .00-.02), a scale effect | S (x10-30) |
| 15 | spell-corrected variants vs same-gate dense_s end-to-end: 0 Holm differences in 24 world-split cells | hash g1 arms: 2-3 Holm wins per A split; safe gate: 0 | F (small) |
| 16 | oracle retrieval ceiling +.22 (T12) / same for all | identical (+.229*, +.226*, +.221*) | no flip |
| 17 | front-learning-speed "wins" in topic-independent worlds (Tctx +.075*, ctl-irrel +.066*; global-ts beats them all) | Tctx +.074*/.071*/.067*, ctl-irrel +.073*/.066*/.064* | no flip (encoder-independent artefact) |
| 18 | CompactRouter u16/q16 equivalent to f64 (TOST +-.005, 8 worlds) with anchors as features | equivalent for hash and hashtok anchors in all 8 worlds, A and B | **NO FLIP** |

Statistics behind the table: rows 1-8 from results/h17_retrieval_*.json compare lists (cluster-bootstrap CIs, Holm over 126) and results/h17_g1.json contrasts (t-CI over 30 seeds, Holm over 32); rows 9-10 from the per-concept paired CIs / TOST in h17_g1.json ctx rows; rows 11-17 from results/h17_route_*.json (paired t over 30 seeds, Holm per family; * = Holm win); full tables results/h17_tables.md T3/T4 and results/h17_summary.txt. A "flip" here is a change of the conclusion's significance/sign per encoder; I did not run a formal difference-in-differences test between encoders (per-seed values are not stored across runs), but the encoder gaps in rows 1-5 are far outside any CI (>10 CI widths).

## 5. Closed-loop routing, numbers (H2000 = mean expected success over 2000 tasks; arm minus the same guarded front; MiniLM / hash / hashtok)

g1 gate (kn injected whenever an anchor is found), spell queries, A held 31-60 (A confirm and B confirm in results/h17_tables.md T3):

| arm | T12 | T12-drift | T48 | T12-noctx | Tctx | ctl-irrel | ctl-oneoff | Tqa |
|---|---|---|---|---|---|---|---|---|
| g1 dense_s | +.022* / -.002 / -.000 | +.023* / .000 / +.002 | +.014* / -.006 / -.003 | +.022* / -.000 / -.000 | +.075* / +.074* / +.070* | +.069* / +.073* / +.071* | +.008 / -.006 / -.003 | +.023* / +.023* / +.023* |
| g1 bm25_s | +.019* / +.009 / +.013* | +.018* / +.010 / +.016* | +.012* / +.001 / +.008 | +.017* / +.009 / +.011* | +.072* x3 | +.067* x3 | +.009 / +.004 / +.002 | +.022* x3 |
| g1 hybz_s | +.020* / +.011 / +.012* | +.023* / +.010* / +.018* | +.017* / +.002 / +.006 | +.022* / +.009 / +.012* | +.075* x3 | +.070* x3 | +.010* / +.004 / +.004 | +.022* / +.022* / +.024* |

Safe gate: all arms are within -.009..+.009 of the front in the four topic-dependent worlds for every encoder (one Holm win in the 3 splits x 4 worlds x 3 encoders x 2 arms grid shown in T3: MiniLM safe|hybz_s T12-drift +.009* in A confirm; otherwise only the Tctx/ctl-irrel front-speed effects). Oracle retrieval +.229*/.207*/.154*/.209* in T12/drift/T48/noctx for every encoder: the headroom is retrieval hit@1, unchanged by the encoder. Controls: ctl-oneoff is within TOST-equivalence of the front for the safe arms (as in h13); ctl-irrel/Tctx "wins" are front-learning-speed effects (global-ts, post-hoc, beats them; unchanged conclusion from h13). With the h13 original floor grid the MiniLM numbers equal h13 bit for bit (results/h17_route_mini_origgrid.json).

Mechanism (T12, A held, safe/g1 vote arms): vote share .16-.22 (MiniLM), .05-.11 (hash), .08-.14 (hashtok); injected paragraph structurally relevant .20-.24 / .04-.18 / .07-.20; anchor accuracy .50-.52 / .19-.41 / .25-.45. The hash dense_s anchor is nearly uninformative (.19 vs chance 1/12 = .08) so its vote arm barely fires.

## 6. CompactRouter vs f64 with anchor features from hash vs real embeddings (item 3)

H2000, pooled A seeds 31-90 (n=60) and B 61-90 (n=30), 8 worlds, front + 6 anchor arms (safe/g1 x dense_s/bm25_s/hybz_s) + 2 open-vocabulary arms (g1 x dense_s/hybz_s with an extra `kd:<doc id>` feature), x 3 encoders; margin +-0.005 pre-declared:
* compact (u16/q16) vs `recommended_router`: **all 54 (arm, encoder, split) cells equivalent in all 8 worlds** (Holm-TOST < .05 and 95% CI inside +-.005). Worst |diff| = .0017 (MiniLM, B, g1|bm25_s), widest 95%-CI end .0034 (hash/MiniLM, B). For the anchor-free front: |diff| <= .0004.
* `CompactRouterG<f64,f64>` vs `recommended_router`: diff exactly 0 in every cell (decision-identical), as in h5.
* Anchor features from hash retrieval (anchor accuracy .19-.45, noisy, and the open-vocabulary `kd:` doc-id feature with ~40k distinct values) do not change this: equivalence is a property of the cell encoding, not of the feature source.
* Disclosed: in the first (original-grid) MiniLM run one of 14 compact cells failed (B, g1|bm25_s, T48: -.0025 [-.0055,+.0004], picks differ from the final extended-grid picks, which pass); the extended grid run is the reported one.
* Caveat: n=30 in B, so CIs are 2x wider than in A; the anchor feature has only 12/48 distinct cluster values (the `kd:` arms are the open-vocabulary stress); this is TOST on expected success in my synthetic worlds, not on real traffic.

## 7. Cost of the real path (item 5)

| item | MiniLM (real) | hash |
|---|---|---|
| model on disk | all-MiniLM-L6-v2 model.onnx 90,405,214 B (86.2 MiB) + tokenizer.json 466,247 B (reference: py/models/bge-base 436,524,393 B = 417 MiB) | 0 |
| load | 0.18 s session create; RSS 49 -> 152 MB (+103 MB), peak 359 MB after a run | 0 |
| per query text, batch 1 (real h10 queries, 56 chars avg) | 2.42 ms mean, p50 2.14, p95 4.12 | 0.67 us (Rust, single thread) |
| per doc paragraph, batch 1 (157 chars avg) | 4.61 ms mean, p50 4.10, p95 8.46 | 1.39 us |
| batch 32 per text (queries / paragraphs) | 2.49 / 5.93 ms (no gain on 2 threads; length padding) | n/a |
| earlier measurements (results/f6_py_latency.json, h10_embmeta_A.json) | 2.45 ms (b32), 2.62 ms (b1) on synthetic f5 texts; 7.64 ms mean single on the loaded h10 run; 49,214 texts in 189 s (3.8 ms/text) | |
| vector bytes | 1536 B (384 x f32) | 1536 B (dim 384) / 256 B (repo default dim 64) |
| dense scan, 6831 docs | ~1.5 ms/query (h13), identical for any 384-d encoder | same |

Latencies: python onnxruntime CPU, intra-op threads 2, incl. tokenisation and mean pooling, shared 16 GB box (+-50-100%); the harness reproduces the saved h10 fixture vectors exactly (max abs diff 0.0). The encode dominates the whole retrieval cost (2.4 ms vs 0.07 ms fusion + 1.5 ms scan); against the end-to-end gain of the real encoder (+.02 H2000 in T12 for the g1 gate, 0 for the safe gate) that is a real price; the hash path is free but delivers none of the gain.

## 8. How much can the earlier hash-based results be trusted? (item 4 conclusion)

* **Trustworthy, encoder-independent:** router-state/compactness equivalence (CompactRouter, bytes, serving paths that do not depend on cosine geometry), determinism and plumbing; the sign of "spell correction helps typo queries"; the oracle-retrieval ceiling; the front-speed artefacts in topic-independent controls; "retrieval quality is the headroom" (oracle +.22 vs real +.00..+.03).
* **Not trustworthy:** any ranking of retrieval levers (dense vs BM25 vs hybrid; RRF/rerank), the hybrid weight, any cosine-scale threshold (admission floor, adaptive drop, dedup radius), the size of a spell or hybrid gain, anchor accuracy, injected-paragraph relevance, and therefore the size of any retrieval-driven routing gain. The shipped HashEmbedding is a weak lexical encoder (below BM25) with case/punctuation sensitivity that can even make spell correction harmful (g1 bridge -.17 MRR).
* One-line rule: hash worlds validate the plumbing, not the retrieval science. Conclusions about retrieval need the real vector path (or a lexical-weighted one such as BM25) and, for deployment claims, real traffic.

## 9. Negatives / caveats (blunt)

1. All routing outcomes and the g1 corpora are synthetic (mine); only retrieval text on h10 A/B and the MiniLM vectors are real; effect sizes scale with the simulator constants (context gain .15/.20/.85, 5% distraction). Only signs and ties are informative.
2. The real-text hash result is partly a design property of HashEmbedding (no idf, whitespace tokens, 384 buckets). A better synthetic encoder (idf-weighted, stopwords, larger dim, char n-grams) would sit between hash and MiniLM; I did not build one (no new component). hashtok shows the lowercasing alone moves MRR .06 -> .12 but not near MiniLM .29.
3. Topic clusters / competence tables are defined by MiniLM geometry for all encoders; a hash-defined world would change anchor labels. Not run.
4. The adaptive-k and admission-floor comparisons are in cosine units tuned for MiniLM; hash optima sit at grid edges (alpha .1 on g1, drop .02, floors 0.0-0.2). Rows 9, 14 are scale artefacts to a large degree.
5. Seeds resample finite pools; t-CIs over seeds are narrow; the real-doc bootstrap is over distinct queries, g1 and routing have no query-cluster bootstrap (A->B replication is the independence check); B has fewer queries (CIs wider; hit@1/F1@3 hybrid gain for MiniLM is not robust there, as in h13).
6. The h13 RSF tuning bug (2 of 7 grid points) is fixed in h17 only; h13's published RSF picks happened to coincide with the full-grid argmax for MiniLM (.5).
7. Latencies are noisy; the real path's latency is python onnxruntime, not a Rust encoder; ort/fastembed remain unbuilt and unaudited and were not touched.
8. The compact-router test uses the picks tuned for the guarded front and a 12/48-valued anchor (plus the `kd:` stress); a larger-scale deployment with different feature statistics could behave differently (h5 already showed lossy encodings fail in other worlds; u16/q16 is the certified one).

## RERUN (deterministic; only timing fields differ)

```
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r3 RAYON_NUM_THREADS=2 H17_THREADS=3; cd /c/Users/mariu/neural-mesh; PY=C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe
md5sum -c results/h10_corpus.md5                       # saved fixtures
cargo build --release --offline --bin h17_retrieval --bin h17_route --bin h17_g1 --bin h17_hashbench
for m in mini hash hashtok; do H17_EMB=$m $CARGO_TARGET_DIR/release/h17_retrieval.exe; done   # ~45 s each -> results/h17_retrieval_<m>.{json,txt}, h17_top_<m>_{A,B}.bin, h17_picks_<m>.json
for m in mini hash hashtok; do H17_EMB=$m $CARGO_TARGET_DIR/release/h17_route.exe; done        # ~9-10 min each -> results/h17_route_<m>.{json,txt}, h17_route_picks_<m>.json (incl. compact_tost)
H17_EMB=mini H17_THRGRID=orig $CARGO_TARGET_DIR/release/h17_route.exe --out results/h17_route_mini_origgrid.json   # reproduces h13_route.json levels bit-for-bit
$CARGO_TARGET_DIR/release/h17_g1.exe                    # ~40 s -> results/h17_g1.{json,log}
$CARGO_TARGET_DIR/release/h17_hashbench.exe             # -> results/h17_hashbench.json
$PY py/h17_latency.py 2                                 # -> results/h17_latency.json (offline onnxruntime, on-disk MiniLM)
$PY py/h17_summary.py > results/h17_summary.txt; $PY py/h17_tables.py > results/h17_tables.md
```

FILES: src/bin/h17_retrieval.rs, h17_route.rs, h17_g1.rs, h17_hashbench.rs, h17_common/{mod.rs,ext.rs} (h13_common copy + embedder switch), py/h17_latency.py, h17_summary.py, h17_tables.py; results/h17_retrieval_{mini,hash,hashtok}.{json,txt,log}, h17_picks_*.json, h17_top_*_{A,B}.bin, h17_route_{mini,hash,hashtok}.{json,txt,log}, h17_route_picks_*.json, h17_route_mini_origgrid.*, h17_g1.{json,log}, h17_hashbench.json, h17_latency.json, h17_summary.txt, h17_tables.md, lessons_h17.md, h_realembed_validation.md. Reused read-only: results/h10_{corpus,texts,emb,inst}_{A,B}.*, h13_world_{A,B}.json, g1_{corpus,texts,emb}_*. Unchanged: Cargo.toml, src/*.rs.
