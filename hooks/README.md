# hooks/: memory-recall hook and router shadow mode (both opt-in, both silent on any problem)

| File | Claude Code event | What it does |
|---|---|---|
| `recall_hook.py` | `UserPromptSubmit` (stdout is injected into the model context) | gate -> project-scoped `MemoryStore.recall` (k<=3) -> relevance floor -> scrubbed block of <=1500 chars, or nothing |
| `route_shadow.py` | `PreToolUse`, matcher `Task\|Agent` | logs `{ts, session, tool_use_id, actual, mesh_pick, rule_pick, hybrid_pick, n_cand, mesh_ms, hybrid_ms}` to `data/lifelong/route_shadow.jsonl`; prints nothing, never blocks |
| `settings-hooks-recall.json` | | the exact `hooks` entries for `~/.claude/settings.json` (merge by hand, the queen owns that file) |
| `measure_recall.py` | | reproduces every number below (`prompts`, `fire`, `latency`, `tost`) |

Toggles: `NEURAL_MESH_RECALL=off`, `NEURAL_MESH_SHADOW=off` (checked before any import; `python -S` start-up measured 33-38 ms on an idle box,
47-55 ms while a build and a benchmark were running). Budgets: recall 1.2 s internal (`NEURAL_MESH_RECALL_BUDGET_S`) under a 3 s Claude Code timeout;
shadow 2.5 s internal under 8 s. Both exit 0 on every path (garbage stdin, missing/corrupt store, timeout, unwritable log) and are tested.
Tests: `python -m unittest hooks.tests.test_recall_hook hooks.tests.test_route_shadow ingest.tests.test_shadow`.

## 1. Recall hook: measured, and NOT shown to help

Design: cheap gate (code-like paths, backticked code-like spans, identifiers, a distinctive project name, "last time / we decided" phrases; slash commands,
acknowledgements and 1-3 word prompts stop here) -> pool = memories of the project the session runs in (`cwd` basename) plus projects named in the prompt
(no cross-project leakage) -> `MemoryStore.recall` -> floor: cosine >= 0.30 and (>= 3 shared tokens that are rare in the pool, or 1 shared identifier-like
token) -> only the memories actually printed are logged with the session id for usefulness credit -> every printed byte re-scrubbed (`ingest.scrub`; a
SECRET-class hit drops the memory, and the whole block is re-checked).

| Measurement | Result |
|---|---|
| (a) latency, end to end (`python -S hooks/recall_hook.py`), 50 real prompts, 3 runs | p50 86-100 ms, p95 123-145 ms, max 170 ms |
| (a) worst case (every prompt passes the gate, cwd = neural-mesh, pool 194), 2 runs | p50 109-120 ms, p95 144-163 ms, max 182 ms |
| scaling | ~75 us per memory in the pool (600: 67 ms, 5k: 367 ms, 10k: 762 ms). Past ~15k memories in ONE project the 1.2 s budget fires and the hook goes silent (embedding cache = future work) |
| (b) fire rate, real prompts as typed (their recorded cwd), n=207 | gate passes 37%; **fires 0/207 (0.0%)**, 0 tokens added |
| (b) same prompts pretending cwd = neural-mesh | fires 3/207 (1.4%), ~88 tokens (chars/4 estimate) when it fires, 1.3/prompt on average |
| (b) sensitivity: no project scoping | fires 46/207 (22%) at an earlier threshold; the hits I read were off-topic, so the scoping stays |
| (c) bench (`MESH_FRONTEND=hook python bench/run.py --cond A,B --tag hook1`, n=120, haiku-4-5, MAX_THINKING_TOKENS=2048) | hook fired on 1/120 prompts (code co29, 0 of 15 private-repo items); B-A overall -0.008 [-0.042,+0.017]; excluding private -0.009 [-0.048,+0.019] (90% CI [-0.038,+0.019], TOST +-0.03 not met); private 0.00 -> 0.00 |
| tokens | B-A -11 [-55,+49] per item (noise: B equals A on 119/120 prompts) |

How to read (c): on 119 of 120 prompts the hook printed nothing, so B and A got the identical prompt and the -0.008 is haiku replicate noise (2 items
lost, 1 won among 119). The one prompt where the hook fired kept its accuracy. The private-repo items (constants in `src/*.rs`) cannot be helped by the
memory store: none of the 606 memories contains them (grep `N_DOMAINS` = 0). Those facts live in source, which is what the earlier BM25 front-end
(bench v2: 0.00 -> 1.00) retrieves; the memory store holds decisions and lessons. Keep rule (TOST +-.03 outside private AND a gain on private): **not
met, so do not rely on this hook**. It is safe (silent unless confident, cannot fail a turn) but on this user's real prompts it almost never fires,
because prompts are mostly directives ("install X", "continue") and memories are long research summaries. Prompt data: 207 real main-session prompts
exist (ingest state + transcripts, 2026-10-03..10, scrubbed), fewer than the 300 asked for.

## 2. Router shadow mode

`route_shadow.py` asks `neural-mesh route` (on a throw-away COPY of the state: current builds bump counters in the file `route` is given, and the nightly
feedback writes the live file) for its pick, candidates = the agents in the router state, features = the same `ingest.extract.features` the nightly
trains on. `rule_pick` is the fixed phase rule of router-eval (`ingest/shadow.py`, first phase keyword wins, default fr-builder). No prompt text is logged.

`python -m ingest shadow-report [--window 50] [--json]`: accuracy of mesh and rule against the ACTUAL `subagent_type`, Wilson 95% CIs, paired
discordance + McNemar exact p, a rolling window, and the verdict. The rule only knows the 8 fr-* agents, so the promotion statistics use the in-domain
dispatches (actual is an fr-* agent); the all-dispatch line is information only (biased towards the mesh).

**Promotion rule (nothing is turned on):** `ELIGIBLE` iff n_in_domain >= 100 and the mesh's Wilson-95% LOWER bound > the rule's accuracy (point
estimate). Only then may the queen let the hook print a one-line suggestion.

Nightly feed (`ingest/meshio.py`): the actual agent is fed as `feedback --label <agent>`; environment failures (turn limit, rate limit/429/overloaded,
worktree block, API error on a re-dispatch) as `--agent X --reward 0 --failure-class env`; `INGEST_MESH_FEED=reward` restores the legacy feed; a binary
that rejects the flags gets the legacy feed per event (`stats.fallback`), so no event is lost. `python -m ingest rebuild-router [--dry-run] [--eval]`
replays all outcomes into a fresh state with this policy (the live state is backed up first). It was run as a dry run only, see the report.

### R122: labels, per-project states and the shadow `hybrid_pick`
* Outcome labels now come from `ingest/labeler.py` (structural: what the leader did NEXT, fix-task starts, interruption phrases): 53/53 vs the evaluator's hand labels in-sample (v1 text cues: 32/53).
* The nightly also feeds one router state per project (`data/lifelong/mesh/projects/<project>.json`, same policy); `rebuild-router` rebuilds them too.
* `hybrid_pick` (shadow only, nothing is printed): the project's own state, the project's declared roster (`<cwd>/.claude/agents/*.md`) as candidates and the phase rule as `--prior` with `--prior-k 500`.
  `shadow-report` prints it next to mesh and rule. Offline prequential replay (`python -m ingest.router_eval`, results/round3_training.md): this ties the phase rule in-domain and beats the
  project-scoped majority baseline overall; the global-state `mesh_pick` is worse than useless (an arm that was never tried wins the exploration bonus). The promotion rule is unchanged (n_in_domain >= 100 and Wilson lower bound > rule).

## 3. codemem hook (code/doc pointers, opt-in, NOT installed)

`codemem hook` (Rust, `src/codemem/`, exe `data/codemem/bin/codemem.exe`, JSON in `settings-hooks-codemem.json`) is a UserPromptSubmit hook that injects at most 5 `file:line` pointers
(~200 tokens) from a BM25 index of the user's repos when the prompt is a code question, else nothing; always exit 0; `NEURAL_MESH_CODEMEM=off`. Measured: hook p95 50 ms end to end; on 241 real prompts it
fires 0.4% (directives again); in the agent-task A/B (haiku, read-only tools incl. Bash search, held n=31) -29% fresh tokens, -32% time, -57% tool calls, accuracy +0.03 (n.s.). Details, limits and the
battery result (private 0.00 -> 1.00; public TOST +-.03 not met) in `bench/codemem/README.md`. Tests: `python -m unittest hooks.tests.test_codemem_hook`.
