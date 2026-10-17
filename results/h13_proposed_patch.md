# H13 proposed patch (NOTHING applied to src/; opt-in only; defaults unchanged; the director decides)

Evidence: results/h13_board_entry.md (doc-level, two independent real-text worlds; end-to-end routing null).

1. Retrieval front-end, if/when a text -> knowledge retrieval helper is exposed (reference implementations: `src/bin/h13_common/{mod.rs,ext.rs}`: `Bm25`, `zscore`, `fuse_z`, `Store`; spell corrector `py/h10_embed.py` Speller):
   - recommended opt-in mode for REAL documentation text: dense (MiniLM cosine) + BM25 (k1 1.2, b .75, lowercase [a-z0-9]+ tokens, 31-word stoplist) fused by z-score, `score = 0.7*z(dense) + 0.3*z(bm25)` (z over all docs of the store), with the optional spell-correction of the query (doc-vocabulary, ed<=1, OOV words only) as a SEPARATE switch (+.015 MRR at 25% typo queries, -.002 on clean ones).
   - cost: fusion step 79 us/query incl. its top-8 selection = net ~+23 us (+1.4%) over the 1.64 ms dense scan + top-8 on 6.8k docs; the 3-8 ms query encode dominates, +1.0 MB BM25 index per 6.8k paragraphs (+10% over the 10.5 MB f32 matrix), no extra model call.
   - do NOT ship RRF, RSF or the rerank-20 variants (RRF -.02 MRR vs z; RSF no better than z; rerank-20 not replicated in world B).
   - corpus caveat to put in the doc string: the gain was measured on long-tailed real documentation (hapax-rich vocabulary); on templated/paraphrase corpora it is ~0 (MRR +.008..+.013, F1@3 -.07) and on queries with very low lexical overlap with their document it is negative (-.03 MRR in the lowest overlap tertile). Keep dense+spell available as the alternative.
2. Routing integration: no change. If a retrieval-anchored vote layer is ever exposed, keep the g16 safe gate as the opt-in default; on real text the ungated (g1) variant gave 2-3x the gain without harm in the one-off control, but its control behaviour on real traffic is unknown. Do not advertise the choice of retrieval variant as a routing lever: end-to-end all spell-corrected variants tie within .009 (0 Holm differences).
3. Nothing for `neural-mesh serve`.
