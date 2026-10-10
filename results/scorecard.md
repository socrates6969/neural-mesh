# Cross-session scorecard

Generated 2026-10-10T06:52:10+00:00 by `python -m ingest scorecard`. Baseline cutoff: **2026-10-10T06:47:30+00:00** (baseline = every turn/dispatch that started before this cutoff (the round-3 changes had not taken effect yet)). Numbers are aggregates over local Claude Code transcripts (leader sessions plus subagent transcripts); no text is included.

## Baseline headline (before the cutoff)

16 sessions, 238 user turns, 493 dispatches (474 with a subagent transcript).

| metric | median | p90 |
| --- | ---: | ---: |
| tokens per user turn, all (leader) | 1.9M | 10.4M |
| tokens per user turn, fresh = in+out+cache write (leader) | 20k | 442k |
| tokens per subagent dispatch, all | 6.9M | 29.0M |
| tokens per subagent dispatch, fresh | 316k | 975k |
| wall seconds per user turn | 114.1 | 1,622.8 |
| wall seconds per subagent dispatch | 701.4 | 3,151.9 |
| tool calls per user turn | 2 | 17 |
| tool calls per dispatch | 39 | 96 |
| Read+Grep+Glob calls per dispatch (searching) | 1 | 4 |
| search-like Bash calls (grep/rg/find/cat/ls/head/tail) per dispatch | 24 | 62 |
| all searching calls per dispatch (Read+Grep+Glob+search-like Bash) | 26 | 65 |
| hook wall ms per turn (Pre/PostToolUse, prompt hooks; max per event group) | 1,272 | 21k |
| Stop-hook ms per turn (max hook per stop) | 842 | 13k |
| tool_use to tool_result gap, Read, ms (n=723) | 38 | 2,113 |
| tool_use to tool_result gap, Grep, ms (n=25) | 1,607 | 2,960 |
| tool_use to tool_result gap, Glob, ms (n=14) | 2,496 | 18k |
| tool_use to tool_result gap, Edit, ms (n=572) | 3,880 | 19k |
| tool_use to tool_result gap, Write, ms (n=4093) | 7,213 | 28k |
| tool_use to tool_result gap, Bash, ms (n=16109) | 7,052 | 43k |

- re-dispatch rate 14.3% of 463 labeled dispatches (15.7% of labeled-dispatch tokens); fix-round rate 2.2% (1.7% of tokens). Labels are the ingest labeler's heuristic (a proxy).
- compaction events: 7 (0.44 per session).
- cache reads are 95.8% of prompt tokens (input + cache write + cache read).
- output-token under-recording check: 0.0% of API messages that carry text or a tool call have output_tokens <= 3 (a streamed placeholder); sampled recorded output tokens exceed a content-length estimate, so no under-recording is detected by this check. Known gaps instead: rejected/aborted requests leave no usage record, thinking tokens are in usage but not in the visible content, and a leader turn excludes its subagents' tokens (counted per dispatch).
- Hook overhead is read two ways: the hook attachment durations above (parallel hooks of one event count once, as their max), and the tool gaps: Read returns in tens of ms (the no-hook floor), so the excess on Edit/Write/Bash is hooks plus permission waits plus real work, not Read-style latency.
- turn wall time is active time (user message to last assistant/tool-result record) and includes waiting on permission prompts; the leader's tokens are per turn, subagent tokens are per dispatch (see the session rows for totals).

## After the cutoff

No sessions after the cutoff yet. Re-run later: this section fills with after-minus-baseline deltas and bootstrap CIs.

## Trend by day (UTC)

| day | sess | turns | disp | tok/turn med (fresh) | tok/disp med (fresh) | s/turn | s/disp | tools/turn | R+G+G/disp | hook ms/turn | redisp % | fix % | compact | cache % | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-10-03 | 5 | 36 | 0 | 1.6M (7,310) | - (-) | 67.5 | - | 2 | - | 3,534 | - | - | 5 | 95.8% |  |
| 2026-10-04 | 0 | 64 | 104 | 2.2M (10k) | 10.6M (508k) | 96.6 | 1,398 | 2 | 0 | 972 | 0.0% | 1.0% | 0 | - |  |
| 2026-10-05 | 0 | 59 | 296 | 2.2M (26k) | 4.9M (287k) | 238.1 | 454 | 3 | 0 | 0 | 19.9% | 2.0% | 0 | - |  |
| 2026-10-07 | 0 | 1 | 0 | 0 (0) | - (-) | 0.0 | - | 0 | - | 0 | - | - | 0 | - |  |
| 2026-10-08 | 1 | 21 | 12 | 939k (344k) | 10.6M (391k) | 135.4 | 1,296 | 2 | 2 | 1,450 | 16.7% | 0.0% | 2 | 95.4% |  |
| 2026-10-09 | 0 | 19 | 26 | 1.0M (150k) | 4.3M (169k) | 115.8 | 968 | 1 | 2 | 0 | 19.2% | 11.5% | 0 | - |  |
| 2026-10-10 | 10 | 38 | 55 | 2.6M (50k) | 6.9M (305k) | 195.7 | 453 | 4 | 1 | 3,046 | 0.0% | 0.0% | 0 | 96.3% |  |

Per-session rows: `data/lifelong/scorecard.jsonl` (git-ignored, derived from private transcripts).
