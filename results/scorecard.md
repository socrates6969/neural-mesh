# Cross-session scorecard

Generated 2026-10-17T01:30:03+00:00 by `python -m ingest scorecard`. Baseline cutoff: **2026-10-10T06:47:30+00:00** (baseline = every turn/dispatch that started before this cutoff (the round-3 changes had not taken effect yet)). Numbers are aggregates over local Claude Code transcripts (leader sessions plus subagent transcripts); no text is included.

## Baseline headline (before the cutoff)

16 sessions, 238 user turns, 493 dispatches (474 with a subagent transcript).

| metric | median | p90 |
| --- | ---: | ---: |
| tokens per user turn, all (leader) | 1.9M | 10.6M |
| tokens per user turn, fresh = in+out+cache write (leader) | 20k | 442k |
| tokens per subagent dispatch, all | 7.1M | 29.7M |
| tokens per subagent dispatch, fresh | 318k | 1.0M |
| wall seconds per user turn | 114.1 | 1,858.2 |
| wall seconds per subagent dispatch | 733.8 | 3,258.8 |
| tool calls per user turn | 2 | 17 |
| tool calls per dispatch | 39 | 98 |
| Read+Grep+Glob calls per dispatch (searching) | 1 | 4 |
| search-like Bash calls (grep/rg/find/cat/ls/head/tail) per dispatch | 25 | 63 |
| all searching calls per dispatch (Read+Grep+Glob+search-like Bash) | 26 | 66 |
| hook wall ms per turn (Pre/PostToolUse, prompt hooks; max per event group) | 1,272 | 21k |
| Stop-hook ms per turn (max hook per stop) | 842 | 14k |
| tool_use to tool_result gap, Read, ms (n=773) | 44 | 1,931 |
| tool_use to tool_result gap, Grep, ms (n=47) | 1,696 | 2,923 |
| tool_use to tool_result gap, Glob, ms (n=14) | 2,496 | 18k |
| tool_use to tool_result gap, Edit, ms (n=602) | 3,880 | 19k |
| tool_use to tool_result gap, Write, ms (n=4148) | 7,070 | 28k |
| tool_use to tool_result gap, Bash, ms (n=16924) | 7,170 | 43k |

- re-dispatch rate 10.3% of 493 labeled dispatches (10.2% of labeled-dispatch tokens); fix-round rate 3.2% (2.7% of tokens). Labels are the ingest labeler's heuristic (a proxy).
- compaction events: 21 (1.31 per session).
- cache reads are 95.8% of prompt tokens (input + cache write + cache read).
- output-token under-recording check: 0.0% of API messages that carry text or a tool call have output_tokens <= 3 (a streamed placeholder); sampled recorded output tokens exceed a content-length estimate, so no under-recording is detected by this check. Known gaps instead: rejected/aborted requests leave no usage record, thinking tokens are in usage but not in the visible content, and a leader turn excludes its subagents' tokens (counted per dispatch).
- Hook overhead is read two ways: the hook attachment durations above (parallel hooks of one event count once, as their max), and the tool gaps: Read returns in tens of ms (the no-hook floor), so the excess on Edit/Write/Bash is hooks plus permission waits plus real work, not Read-style latency.
- turn wall time is active time (user message to last assistant/tool-result record) and includes waiting on permission prompts; the leader's tokens are per turn, subagent tokens are per dispatch (see the session rows for totals).

## After the cutoff

3 sessions, 107 turns, 136 dispatches.

| metric | stat | after - baseline | 95% CI (session bootstrap) | n base / n after | sessions base / after |
| --- | --- | ---: | --- | --- | --- |
| tokens per turn (all, incl. cache read) | median | +834,687.5 tok | [-2,175,433.0, +7,911,061.5] | 238 / 107 | 13 / 6 |
| tokens per turn (all, incl. cache read) | p90 | +6,473,633.8 tok | [-9,558,657.2, +16,514,708.3] | 238 / 107 | 13 / 6 |
| tokens per turn (fresh: in+out+cache write) | median | +585.5 tok | [-117,643.0, +45,215.0] | 238 / 107 | 13 / 6 |
| tokens per turn (fresh: in+out+cache write) | p90 | +156,563.2 tok | [-1,459,893.5, +499,431.4] | 238 / 107 | 13 / 6 |
| tokens per dispatch (all) | median | -1,246,003.5 tok | [-3,519,505.5, +948,971.0] | 474 / 135 | 5 / 3 |
| tokens per dispatch (all) | p90 | +751,998.4 tok | [-68,354,829.4, +70,417,030.0] | 474 / 135 | 5 / 3 |
| tokens per dispatch (fresh) | median | -30,779.5 tok | [-47,394.0, +56,588.5] | 474 / 135 | 5 / 3 |
| tokens per dispatch (fresh) | p90 | +152,696.1 tok | [-1,268,683.2, +5,545,056.0] | 474 / 135 | 5 / 3 |
| wall per turn | median | +57.1 s | [-108.2, +609.2] | 238 / 107 | 13 / 6 |
| wall per turn | p90 | +2,982.9 s | [-2,588.4, +13,765.5] | 238 / 107 | 13 / 6 |
| wall per dispatch | median | -220.4 s | [-541.2, +483.1] | 474 / 135 | 5 / 3 |
| wall per dispatch | p90 | +2,650.4 s | [-11,064.1, +10,128.3] | 474 / 135 | 5 / 3 |
| tool calls per turn | median | +4.0  | [-2.0, +13.0] | 238 / 107 | 13 / 6 |
| tool calls per turn | p90 | +15.1  | [-27.6, +24.8] | 238 / 107 | 13 / 6 |
| tool calls per dispatch | median | -3.0  | [-21.0, +19.0] | 474 / 135 | 5 / 3 |
| tool calls per dispatch | p90 | +17.9  | [-84.4, +95.4] | 474 / 135 | 5 / 3 |
| Read+Grep+Glob per dispatch | median | +3.0  ** | [+1.0, +4.0] | 474 / 135 | 5 / 3 |
| Read+Grep+Glob per dispatch | p90 | +10.6  ** | [+2.6, +16.0] | 474 / 135 | 5 / 3 |
| search-like Bash (grep/rg/find/cat/ls...) per dispatch | median | -4.0  | [-18.0, +13.0] | 474 / 135 | 5 / 3 |
| search-like Bash (grep/rg/find/cat/ls...) per dispatch | p90 | +19.0  | [-21.0, +67.7] | 474 / 135 | 5 / 3 |
| all searching per dispatch | median | +0.5  | [-14.0, +21.0] | 474 / 135 | 5 / 3 |
| all searching per dispatch | p90 | +23.2  | [-20.5, +75.8] | 474 / 135 | 5 / 3 |
| hook wall per turn | median | +93.5 ms | [-16,743.0, +23,056.5] | 238 / 107 | 13 / 6 |
| hook wall per turn | p90 | +30,156.3 ms | [-47,428.5, +74,228.6] | 238 / 107 | 13 / 6 |

A row in bold has a CI that excludes 0. Units inside a session are correlated, so sessions are resampled, not turns.

## Trend by day (UTC)

| day | sess | turns | disp | tok/turn med (fresh) | tok/disp med (fresh) | s/turn | s/disp | tools/turn | R+G+G/disp | hook ms/turn | redisp % | fix % | compact | cache % | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-10-03 | 5 | 36 | 0 | 1.6M (7,310) | - (-) | 67.5 | - | 2 | - | 3,534 | - | - | 5 | 95.8% |  |
| 2026-10-04 | 0 | 64 | 104 | 2.2M (10k) | 10.6M (508k) | 96.6 | 1,398 | 2 | 0 | 972 | 0.0% | 0.0% | 0 | - |  |
| 2026-10-05 | 0 | 59 | 296 | 2.2M (26k) | 4.9M (287k) | 238.1 | 454 | 3 | 0 | 0 | 14.2% | 4.1% | 0 | - |  |
| 2026-10-07 | 0 | 1 | 0 | 0 (0) | - (-) | 0.0 | - | 0 | - | 0 | - | - | 0 | - |  |
| 2026-10-08 | 1 | 21 | 12 | 939k (344k) | 10.6M (391k) | 135.4 | 1,296 | 2 | 2 | 1,450 | 25.0% | 0.0% | 2 | 95.4% |  |
| 2026-10-09 | 0 | 19 | 26 | 1.0M (150k) | 4.3M (169k) | 115.8 | 968 | 1 | 2 | 0 | 19.2% | 3.8% | 0 | - |  |
| 2026-10-10 | 11 | 94 | 145 | 2.0M (18k) | 6.2M (292k) | 104.8 | 435 | 4 | 2 | 4,148 | 2.1% | 9.7% | 14 | 95.8% |  |
| 2026-10-11 | 0 | 15 | 27 | 4.5M (491k) | 14.1M (623k) | 2,762.7 | 2,518 | 6 | 5 | 0 | 0.0% | 0.0% | 0 | - | after |
| 2026-10-12 | 0 | 34 | 19 | 5.0M (60k) | 5.2M (291k) | 348.5 | 790 | 9 | 3 | 0 | 0.0% | 15.8% | 0 | - | after |
| 2026-10-15 | 2 | 2 | 0 | 0 (0) | - (-) | 0.0 | - | 0 | - | 36 | - | - | 0 | - | after |

Per-session rows: `data/lifelong/scorecard.jsonl` (git-ignored, derived from private transcripts).
