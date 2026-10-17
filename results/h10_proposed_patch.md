# H10 proposed patch (nothing applied to src/)

No change to src/ is proposed or needed by H10: CompactRouter and the serve fix survived the real-text replay unchanged.

Optional, opt-in only (not applied; the director decides):
1. Retrieval front-end, if/when a text->knowledge retrieval helper is exposed: spell-correction on the query (doc-vocabulary, ed<=1, OOV words only) is worth +.015 MRR at 25% typo queries on real doc text but costs -.002 MRR on clean queries (Holm-significant loss), so it should be a switch, not a default; and z-score dense+BM25 fusion (alpha .7) is the larger lever on real text (+.03..+.05 MRR) with BM25 scoring at ~3 us/query. Reference implementations: `src/bin/h10_common/mod.rs` (Bm25, zscore) and `py/h10_embed.py` (Speller).
2. `neural-mesh serve`: nothing new beyond g6's note that the per-request `stat` in `Server::refresh()` is the next bottleneck (NEURAL_MESH_SERVE_STAT_MS opt-in already exists). h10 did not re-measure it.
