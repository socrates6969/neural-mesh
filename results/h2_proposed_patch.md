# H2 proposed patch (R4-A2) — nothing applied to src/

Evidence: results/h2_board_entry.md (H2-B/C). Synthetic-until-validated-on-real-traffic.

## 1. Do NOT add (rejected by the end-to-end test)
- Reciprocal Rank Fusion (k=60) and Relative Score Fusion between the dense and tf-idf legs of the knowledge-vote similarity.
  No end-to-end gain over spell-only dense (|d| <= .01 vs spell-only in nearly all cells (one C2 sib16 Holm loss -.009 under the g1 gate), no Holm win in 9 worlds x 4 corpora), 5x the per-task loop cost
  (33-36 us vs 7-8 us), and rank-only fusion has no absolute scale for the vote layer's "no neighbour" decision.
- A top-20 cheap reranker (token overlap / BM25 / small logistic) on the vote's neighbour list: ~12 scorer calls/task, no end-to-end gain
  (logistic: +.028 doc-level F1@3 but -.001 sib16 end-to-end under the safe gate); overlap/BM25 rerank lose rand16 (-.014..-.029 vs spell-only).

## 2. Optional simplification (only together with the g16 conservative gate, results/g16_proposed_patch.md)
If the g16 "G-post+g1" gate (P(Beta posterior of the anchor's best agent > front reference) >= .8 AND the g1 window gate, `kn:<anchor>` injected only when the
vote is open) is adopted behind an opt-in flag, the similarity can be MiniLM on spell-corrected text alone (thr .75, k 5, min_n 2): it ties the tf-idf hybrid there
(F4: sib16 -.002, rand16 +.008, sib64 +.001, controls -.003..+.002, all n.s., corpus A; B/C2/D2: |d| <= .011, all n.s.) and drops the sparse leg (sparse vectors ~320 B/task, tf-idf build).
Under the old g1 gate (inject-always) keep the hybrid: it is +.023* over spell-only in sib16 but -.029! in rand16 (world-dependent), and every g1-gate variant
loses ctl-irrel / ctl-oneoff (-.014..-.028!), so the g1 gate should stay opt-in only.

## 3. Defaults unchanged
No default changes are proposed. Spell-correction remains the only retrieval-side lever with a consistent end-to-end effect (g1-gate sib16 +.034*, drift +.017*
vs raw dense), and it only matters when the query stream has typos (25% in this corpus).
