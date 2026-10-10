# Round 3 training (trainer, 2026-10-10): labels, router, SONA, LLM data

Every number names its command. Nothing here is promoted; the gates decide. Honest short version: **labels improved a lot (in-sample), the learned routers still do not beat the phase rule or the majority baseline where it matters, SONA's persisted-pattern path cannot fire, and the SFT backlog is being cleared across sessions with a decoy contrast.**

## 1. Label quality (everything downstream)

Command: `python -m ingest.labeler validate` (hand labels: `ingest/tests/fixtures/hand_labels.json`, the evaluator's labels from the freight-recovery run dirs' status.md / mailbox.md; 53 usable = passed / defects / fix-round / re-dispatch / blocked; `pending` and `n/a` rows excluded, as in `validate_labels.py`).

What changed (`ingest/labeler.py`, wired into `ingest/pipeline.py`; `INGEST_LABELER=v1` restores the old classifier): v1 read text cues in the agent's FINAL report. For a chain (a dispatch plus its SendMessage continuations) that report is the same for every segment, and most outcomes are only visible from what the leader did NEXT. v2 uses:
* the next message to the same agent ("you hit your 100-turn limit", "terminated by an API rate limit (HTTP 429)") -> this segment was cut off = re-dispatch (env);
* a builder fix message of the matching origin (tester defects D-n, reviewer findings, not CI) after a builder / tester / reviewer segment -> that segment found defects;
* the segment's own message starting as a fix task ("Builder fix round 3", "CI diagnosis", "Small follow-up ... CI result") -> fix-round;
* report text only on the LAST real segment, and only interruption-shaped phrases for env failures ("rate limiting" in a product report is not an env failure), explicit verdict tokens (`TEST_BLOCK`, `DEFECTS_FOUND`, `CHANGES_REQUESTED`, `Verdict: REJECTED`), `SHIP_BLOCKED` / permission-classifier denials for blocked; `NOT promoted` / `REJECTED` are recorded as `verdict_gate_negative` signals, not as defects of the work;
* queued notes to a running agent ("Note from leader ... no reply needed") are not units of work and no longer carry the chain's report.
Not used: the run dirs' mailbox.md / status.md directly (dates only, no times, 6 runs; the same information is in the leader's messages). Not done: loosening the taint rules (see section 4).

| | v1 (text cues) | v2 (structural) |
|---|---|---|
| exact agreement with 53 hand labels | **32 / 53** | **53 / 53** (in-sample) |
| success vs not | 33 / 53 | 53 / 53 |

Confusion, rows = hand label, columns = predicted (success / defect / fix-round / re-dispatch / blocked):

| hand \ v1 | success | defect | fix | re-disp | blocked | | hand \ v2 | success | defect | fix | re-disp | blocked |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| success (25) | 15 | 4 | 6 | 0 | 0 | | success (25) | 25 | 0 | 0 | 0 | 0 |
| defect (9) | 5 | 3 | 1 | 0 | 0 | | defect (9) | 0 | 9 | 0 | 0 | 0 |
| fix-round (11) | 2 | 0 | 9 | 0 | 0 | | fix-round (11) | 0 | 0 | 11 | 0 | 0 |
| re-dispatch (7) | 2 | 0 | 0 | 5 | 0 | | re-dispatch (7) | 0 | 0 | 0 | 7 | 0 |
| blocked (1) | 1 | 0 | 0 | 0 | 0 | | blocked (1) | 0 | 0 | 0 | 0 | 1 |

**Read this carefully: 53/53 is IN-SAMPLE.** I wrote the rules while reading those 53 labels (the first version scored 50/53, two of the three misses fixed by making env-resume detection skip queued notes and by making builder-defect detection chain-local; one by narrowing the text-count rule). The rule set is small and each rule is a general statement about the leader's messages, but 53/53 is the ceiling of an overfit-prone procedure, not an estimate of accuracy. The one out-of-sample look I could do: 20 randomly drawn non-hand-labelled non-success labels (10 re-dispatch, 2 defect, 6 fix-round, 2 blocked), read by me against the message and report tail: 16 plausible, 4 doubtful (both `defect` rows and both `blocked` rows: a tester continuation flagged by a later fix message, a report line "1 failed overall" about another team's test, and two `needs your approval` phrases). That is not independent labelling; the number to trust is the next hand-labelled batch.

Effect on the 563 recorded outcomes (`python -m ingest.labeler relabel --write`, backup `data/lifelong/outcomes.v1.jsonl`): success 397 -> 475, defect 68 -> 11, fix-round 30 -> 23, re-dispatch 66 -> 51, blocked 2 -> 3. 113 rows changed: defect->success 64 (domain words such as REJECTED), fix-round->success 14, re-dispatch->success 17 (a product report that mentions rate limits), success->fix-round 8, success->defect 6, success->re-dispatch 2, success->blocked 1, fix-round->defect 1.

## 2. Router

Router state: backed up as `data/lifelong/mesh/router_state.json.pre-rebuild-r122` (and the older `.pre-rebuild`), then `python -m ingest rebuild-router` (label/env feed, v2 outcomes). Coordination note: the live state before the rebuild held a `merge-state` of an external state (9 extra agents: Plan, builder-gp, builder-isolated, distiller, docs-gp, planner, reviewer, shipper, tester; 607 vs 550 feedbacks). A replay cannot reproduce that; those agents are only in the backup (`neural-mesh merge-state --into LIVE --from BACKUP` restores them). New: one router state per project (`data/lifelong/mesh/projects/<project>.json`, 11 of them), fed by the same nightly policy.

**The label feed ignores quality outcomes.** In `--label` mode every non-env row is fed as "the leader's actual agent" regardless of success / defect / fix-round, so better outcome labels only change which rows go to the env class (66 -> 51; 17 false env rows removed). That is why section 1's gain does not move router accuracy:

Prequential replay (`python -m ingest.router_eval --outcomes <v1|v2 outcomes>`; predict each dispatch with `route` on the state built from everything earlier, then feed it; chronological and leak-free; env rows are not scored; Wilson/McNemar via `ingest/shadow.py`). "maj" = prequential majority-so-far baseline in the same candidate scope. "fr" = dispatches whose actual agent is an fr-* role (the only place the phase rule can be right), n = 22.

| feed | state / candidates | prior (k) | all dispatches (v1 labels) | all (v2) | maj | fr (v2) | phase rule |
|---|---|---|---|---|---|---|---|
| label | global / all seen (today's hook) | - | 258/425 = .607 | 265/439 = .604 | 361/439 = .822 | 5/22 = .227 | 19/22 = .864 |
| label | per-project state | - | - | 235/303 = **.776** | 222/303 = .733 | 10/22 = .455 | 19/22 |
| success-only imitation (defect/fix penalised) | per-project | - | - | 209/303 = .690 | 222/303 | 6/22 = .273 | 19/22 |
| label | per-project + roster | 20 | - | 233/305 = .764 | 222/305 | 15/22 = .682 | 19/22 |
| label | per-project + roster | 100 | - | 236/305 = .774 | 222/305 | 18/22 = .818 | 19/22 |
| label | per-project + roster | **500** (hybrid) | - | 237/305 = **.777** | 222/305 | **19/22 = .864** | 19/22 |
| label | per-project + roster | 5000 | - | 237/305 = .777 | 222/305 | 19/22 = .864 | 19/22 |

Paired exact McNemar (mesh-only-right : other-only-right; `re_v2_grid3.json` in the trainer scratch, reproduced by the command above with `--variants grid2`):
* per-project state vs majority-so-far, all dispatches: **16 : 3, p = 0.0044**. The per-project state is a real improvement over "predict the project's most common agent" and a very large one over the global state the shadow hook uses today (global: .604 vs maj .822, i.e. BELOW the trivial baseline).
* per-project state vs the phase rule, fr-* dispatches: 1 : 10, p = 0.012. **The learned router loses to the rule in-domain, significantly.**
* hybrid (project state + declared roster as candidates + phase-rule `--prior`, `--prior-k 500`) vs the rule: 0 : 0 (it IS the rule on freight-recovery, plus the learned per-project router elsewhere); vs majority-so-far on all dispatches **20 : 5, p = 0.0041**. Intermediate k values interpolate between the mesh (.455) and the rule (.864); no k beats the rule. The "learned correction" adds nothing measurable at n = 22.
* success-only feed: lower on every axis (it is scored on imitation, which it is not trying to maximise; no outcome-based metric exists that could reward it with this much data).
* Why the live shadow log looks so bad (`python -m ingest shadow-report`: mesh 0/21 in-domain, 0/53 all): with the whole roster as candidates an arm that was never tried wins the exploration bonus (the old live state picked `Plan` for all 25 fr-* tasks, the per-project state with the full roster picks `fr-ci-diagnoser`, never used); the prequential replay avoids it because candidates are "seen so far". A prior (or only offering tried arms) is what removes it.

Gate (analogue of mesh-coder's: significant paired win with no significant loss on another axis, tuned then confirmed): **not met, nothing promoted or enabled.** There are no seeds in real traffic; n_in_domain = 22 (promotion needs >= 100 and a Wilson lower bound above the rule); k = 500 was chosen from the same replay it is reported on (exploratory, not tuned/confirmed). What was committed: `ingest/router_eval.py` (the harness), per-project states in the nightly feed / `rebuild-router`, and a **shadow-only** `hybrid_pick` field in `hooks/route_shadow.py` + `shadow-report` (project state, `<cwd>/.claude/agents/*.md` roster, phase-rule prior k = 500), so the promotion rule can be evaluated on live traffic; nothing is printed or injected.

## 3. SONA / what `hooks route` reads

`@claude-flow/cli` dist/src/mcp-tools/hooks-tools.js `hooksRoute`, in order:
1. AgentDB bridge: hard-disabled on win32 (plugin-medic) -> never answers.
2. SONA persisted patterns (`memory/sona-optimizer.js getRoutingSuggestion`, source `sona-pattern`): used only if confidence >= 0.6 AND the agent is one of 10 ruflo role names (coder, tester, reviewer, architect, researcher, optimizer, debugger, documenter, security-architect, performance-engineer): **an fr-* name is never eligible**. Written only by `hooks_intelligence_trajectory-end` -> `processTrajectoryOutcome`, and only when the memory DB is initialised (`claude-flow memory init`; otherwise "Database not initialized", `sonaUpdate:false`). A pattern is keyed by the task's first <= 10 keywords + agent; confidence starts 0.5 and gains 0.1*(1-c) per success; a route needs overlap x confidence >= 0.6. Real tasks never repeat the same keyword set, so confidence stays at 0.55.
3. `routeTaskLocal`: a semantic index (default embedder `hash`; `CLAUDE_FLOW_ROUTER_EMBEDDER=minilm` is supported) over static TASK_PATTERNS + `learned-<agent>` patterns rebuilt from `.claude-flow/routing-outcomes.json` (written by `hooks_post-task` with task + agent + success + quality; learned patterns need support >= 2, reliability >= .75, similarity > 0.4 between the whole task and ONE keyword). Otherwise keyword fallback, default `coder`.
The previous bench (`bench/ruflo_train.mjs`) fed ruflo through one-shot CLI calls (post-task with ruflo role names + `hooks intelligence --train`): a trajectory cannot survive between two CLI processes, and `--train` feeds neither store above.

`bench/ruflo_train2.mjs` calls the same tool handlers in-process in a fresh sandbox (after the supported `memory init`), trains from the v2-labelled freight-recovery dataset (`bench/build_ruflo_dataset.py` -> `bench/data/ruflo_dispatch_v2.jsonl`, 72 dispatch+continuation rows, 7 of the 8 roles present), chronological 70/30 split: 50 train / 22 test. Majority baseline on test (fr-builder): 8/22 = **.364**; phase rule on test 17/22 = **.773**. Result file `bench/ruflo_train2_result.json` (descriptive, NOT gated; exact one-sided binomial vs the majority baseline):

| variant | embedder | test before -> after | after k/22 | p vs majority | train before -> after | route changes on test |
|---|---|---|---|---|---|---|
| base (no training) | hash | .227 | 5 | .94 | - | - |
| post-task, native fr-* names, v2 success flags | hash | .182 -> .136 | 3 | .996 | .50 -> .54 | 6 |
| post-task, every row a success (imitation) | hash | .182 -> **.273** | 6 | .87 | .50 -> .64 | 9 |
| post-task, ruflo role names (old convention) | hash | .182 -> .136 | 3 | .996 | .50 -> .54 | 6 |
| SONA trajectories (ruflo names, v2 success) | hash | .182 -> .182 | 4 | .98 | .50 -> .50 | **0** |
| post-task native + SONA | hash | .182 -> .136 | 3 | .996 | .50 -> .54 | 6 |
| base | minilm | .364 | 8 | .58 | - | - |
| post-task native / imitation / + SONA | minilm | .364 -> .273 | 6 | .87 | .60 -> .58 / .78 | 15 / 13 / 15 |
| SONA trajectories | minilm | .364 -> .364 | 8 | .58 | .60 | **0** |

(The `before` column of identical untrained sandboxes differs by one row, .18 vs .23: ties in the native index are not deterministic. Train accuracy is route accuracy on the training rows.)
Findings: (a) after 50 trajectories SONA holds 50 patterns, **max confidence 0.55, 0 patterns >= 0.6**, so its path changed 0 of 22 test routes: it is dead for routing as built, whatever is trained into it. (b) The `learned-<agent>` path does change routes (6 to 15 of 22) and raises TRAIN accuracy (.50 -> .64 hash, .60 -> .78 minilm, imitation) but test accuracy goes down or stays flat: it overfits 50 rows. (c) **Nothing beats the majority baseline (best .273 hash; minilm untrained is already at the baseline .364), and nothing is near the phase rule (.773).** Quality-aware flags (fail rows as success=false) were worse than imitation here, as in section 2. Plainly: with this data the ruflo learning layer cannot be trained into a better `hooks route` than the phase rule or even a constant `fr-builder`.

## 4. LLM data

Problem found (data/lifelong/runs.jsonl, first real run): the local judge budget (~400 calls) was spent in time order on dispatch reports, so the 169 user-turn candidates never got a call; all 203 SFT rows come from ONE session although 8 sessions exist; and a dispatch whose outcome row had already been written was never reconsidered for SFT (the old code looked for an SFT row only when it wrote the outcome row), so every row skipped for lack of budget (241 in run 1) was lost for good. The local qwen judge also admits ~40% of decoys (judge-smith's calibration).

Changes (`ingest/schedule.py`, `ingest/pipeline.py`; tests `ingest/tests/test_schedule.py`):
* SFT candidates are judged first, round-robin over sessions (latest first) with user turns and agent reports alternating, newest first; a small budget now spreads over sessions and sources.
* Candidates are re-examined on every run (retry of budget-starved dispatches; a pending row keeps its cached real grade). Only `kind=dispatch` rows are candidates: a continuation's stored report is the chain's final report, not an answer to that message.
* Decoy contrast: a row needs real >= 0.7 AND decoy_score <= real - 0.3, where the decoy is the answer of the MiniLM-nearest other candidate request (response cosine < 0.85, same construction as `llm/calibrate.py`) graded by the same local judge against the same request. No encoder / no decoy -> admitted on the real grade, row says `decoy_score: null`; decoy not graded for lack of budget -> pending, not rejected. Rows carry `decoy_score`, `decoy_margin`; drops are counted as `sft_decoy_not_rejected`.
* Nightly (`llm/scripts/run_nightly.ps1`): `-JudgeBudget` (local, free) 600 -> 800 and now ONLY feeds the ingestor; new `-PairBudget 120` feeds the paid Haiku gate (before, the one number 600 went to both, i.e. up to 600 paid calls a night against the README's 120).

Measured (`python -m ingest run --judge-budget 450`, 843 s, 255 local judge calls + 160 cache hits, last row of data/lifelong/runs.jsonl): **sft.jsonl 203 -> 301 rows, sessions 1 -> 11**; user_turn rows 0 -> 65. Per (session, source) after the run: 8f3352b3 agent_report 216 + user_turn 54; df94bcb2 9; 088e806e 5; b538d3d8 6; 0474a764 3; 4577438c 2; ad9c19e8 2; efd71ab0 1; 277c4c8d 1; 33cd032a 1; dbfa151b 1. So >= 3 sessions is met, but 8f3352b3 is still 270 of 301 rows (the other sessions have few qualifying items, 1-9 each).

Decoy contrast: 164 candidates were decoy-checked; 98 admitted, **66 rejected (40%) because the judge scored the nearest-but-wrong answer within 0.3 of the real one** (`dropped.sft_decoy_not_rejected`). Of 159 decoy grades, 68 (43%) scored >= 0.7, matching the ~40% decoy acceptance judge-smith measured, so the old absolute-score rule alone would have admitted those rows. The 203 older rows were admitted without the check and were NOT re-screened (about 203 more local calls). Other drops: tainted 5 agent reports + 15 turns, secret 7 + 6, 1 negative-feedback turn. `sft_meta.jsonl` was rewritten, so the next `llm nightly` has multi-session held-out data. Not done on purpose: relaxing the taint rules (149 of ~450 agent reports in the big session were dropped as `sft_tainted` in the first run), a secrets-policy decision.

## Reproduce
* labels: `python -m ingest.labeler validate`, `python -m ingest.labeler relabel [--write]`; tests `python -m pytest ingest/tests`
* router: `python -m ingest.router_eval --outcomes data/lifelong/outcomes.jsonl [--variants grid2] --json out.json`; rebuild `python -m ingest rebuild-router [--dry-run] [--eval]`
* ruflo: `RUFLO_PKG_SRC=<@claude-flow/cli/dist/src> node bench/ruflo_train2.mjs --all` (dataset: `python bench/build_ruflo_dataset.py`)
