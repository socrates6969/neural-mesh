# Learning curve of the local model on real session data (2026-10-12)

**Question:** does the local model (Qwen2.5-Coder-3B-Instruct + QLoRA, `llm/`) get better as it trains on more real data?

**Data.** `data/lifelong/sft.jsonl`: 372 instruction/response pairs extracted by the ingestor (`ingest/`) from 11 real
Claude Code sessions recorded between 2026-10-03 and 2026-10-12, scrubbed for secrets and personal data before storage
(sha256 prefix of the file used: `68b10770c835401c`). The rows are not published (private work transcripts); this
directory holds numbers only.

**Method.** `python -m llm learning-curve` (`llm/learning_curve.py`). Every real adapter and the base model are scored on ONE
common held-out set: 75 rows from the 6 most recent sessions (63 paraphrase clusters), excluding every row any compared adapter
was trained on. Metric: per-token negative log-likelihood of the held-out responses (lower = the model predicts the real
later work better). Interval: 95% cluster bootstrap of the per-example improvement over the base, 4,000 resamples.
All three adapters: fresh LoRA from the base, 2 epochs, same hyper-parameters.

| Model | Real session rows trained | Anchor rows | Held-out perplexity | Improvement vs base (95% CI) |
|---|---:|---:|---:|---|
| base | 0 | 0 | 20.20 | - |
| 20261010-033411 | 163 | 0 | 15.51 | +0.277 [+0.251, +0.306] |
| 20261011-014210 | 277 | 0 | 14.37 | +0.364 [+0.324, +0.409] |
| 20261012-160854 | 297 | 288 | 14.10 | +0.384 [+0.341, +0.433] |

Source: `2026-10-12-learning_curve.json`; training sizes: `2026-10-12-adapters.jsonl`.

**General reasoning (deterministic battery, 20 items per category, fixed scorer, `2026-10-12-eval-20261012-160854.json`).**
Without anchors (20261011-014210): math 3 vs base 8, physics 1 vs 8, code 17 vs 18, research 6 vs 9.
With self-distillation anchors (20261012-160854): math 10 vs 8, physics 11 vs 8, code 20 vs 18, research 4 vs 9.

**What this shows.** More real session data lowered held-out loss on unseen later sessions: 163 -> 277 rows moved the
improvement interval from [0.251, 0.306] to [0.324, 0.409] (non-overlapping). Anchors removed the math/physics/code
forgetting. 277 -> 297 rows (+20) is not distinguishable.

**What it does not show.** Held-out loss measures prediction of the next sessions' text, not answer quality. The paired
Claude-Haiku judge has not shown a significant quality win (20261011-014210: win rate 0.533, CI [0.422, 0.639], n = 60;
20261012-160854: 27 of 60 items judged before the daily judge budget ran out). Research recall still regresses. No model
has passed the promotion gate; production remains the base model. The comparison is observational across nightly rounds
(more data and later data move together; the last row also adds anchors).
