# llm/ — local LLM (owner: llm-smith)

Everything runs on this box; nothing is sent to external services. Venv: `llm/.venv` (git-ignored), Python 3.14,
torch 2.11.0+cu128 (sm_120 verified with a real fp16 matmul: 48 TFLOPS, max err vs CPU 0.13 on |x|<=381),
transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2 (nf4 OK), accelerate 1.15.0, datasets 5.1.0, onnxruntime (MiniLM).
Install: `uv venv llm/.venv --python 3.14` then `uv pip install -c llm/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match torch transformers peft bitsandbytes datasets accelerate safetensors onnxruntime`.

Commands (run from the repo root with `llm/.venv/Scripts/python -m llm ...`):
`bench` (base choice), `train`, `eval`, `promote`, `status`, `frontend-eval`, `nightly` (train+eval+promote under one lock).
`--synthetic` uses a LABELLED synthetic sample instead of `data/lifelong/sft.jsonl`.

* Base: Qwen2.5-Coder-3B-Instruct, chosen by `bench` (models/bench.json): QLoRA nf4, r=16 all proj, seq 1024, bs 1:
  1.5B 0.85 s/example, 4.17 GB peak; 3B 1.44 s/example, 5.37 GB peak (rule: largest that fits 6 GB and <= 3x slower).
* Train: continual LoRA from the production adapter on new rows + replay of already-trained rows (weighted by the
  ingestor's `replay_weight`); held-out = the ingestor's cluster-aware `heldout` flag (sft_meta.jsonl), else latest sessions.
* Eval gate (2026-10-10, replaces the absolute local-judge battery, which accepted 40% of decoys): PAIRWISE Claude Haiku judge
  (`llm/pairjudge.py`, `claude -p --model haiku`, JSON output, MAX_THINKING_TOKENS 1024, no tools). Candidate vs production
  answers on held-out requests, each pair judged in BOTH orders (first order randomised by a content-hash coin); a win needs both
  orders to agree, else tie. Gate = calibrated judge AND >= 40 judged items AND cluster-bootstrap 95% CI lower bound of the win
  rate (win 1 / tie .5 / loss 0) > 0.5 AND no regression in any deterministic battery category (bench/battery.json, $0:
  per-category accuracy drop <= 0.10 and sign-test p > 0.10, pooled drop <= 0.05) AND held-out NLL not significantly worse
  AND >= 3 clusters. Results cached by content hash (`models/pair_cache.jsonl`); hard per-night budget (default 120 calls,
  `models/pair_budget.json`, `--judge-budget N`); token usage from the CLI's JSON usage fields.
* Calibration before trust: `python -m llm calibrate` (<= 200 calls). The judge must prefer the real answer over the
  nearest-but-wrong decoy in >= 90% of >= 40 pairs, both orders, AND agree with the deterministic labels (bench math/physics/code
  pairs: correct vs wrong answer) in >= 85%. Result in `models/pair_calibration.json`; without a passing record for the current
  rubric version + judge model the gate stays closed. The local qwen judge remains only the cheap pre-filter for ingest SFT rows,
  behind the $0 structural check (`ingest/structural.py`: empty / refusal / truncation / repetition / parse).
* Promote: merge on CPU -> llama.cpp `convert_hf_to_gguf.py` (official release tarball in models/tools/) -> q8_0 GGUF ->
  `ollama create mesh-llm:<ver>` -> `ollama cp` to `mesh-llm:latest`. (Ollama 0.40's own safetensors import rejects Qwen2.)
* Front-end (feature 1): policy in `src/frontend.rs`, CLI `neural-mesh frontend` (default OFF). Signals from
  `llm/frontend.py`: verified cache, MiniLM max-cosine OOD guard, Ollama `logprobs` confidence, answer length, guarded kNN tool route.
  Details, consumers, ladder and gates: section 'Front-end' below.
* Nightly (provided, not installed): `llm/scripts/run_nightly.ps1` (runs the ingestor first, then train/eval/promote).

## Pairwise judge: measured 2026-10-10 (Claude Haiku via `claude -p --model haiku` = claude-haiku-4-5; every number from `models/pair_calibration.json` / `adapters/20261010-033411/eval.json`)
* Calibration PASSED (148 calls, 333,976 input + 97,064 output tokens, $0.82): decoy test 44/44 (real answer preferred in both orders; n = 44 pairs from 203 SFT rows, decoy = answer to the MiniLM-nearest different request with response cosine < 0.85); deterministic agreement 30/30 (11 math + 16 physics + 3 code bench pairs: harness-correct Claude answer vs harness-wrong local answer, both orders). Position bias (same letter in both orders) 0.0 in calibration.
  Caveat: both sets are easy by construction (a wrong-topic answer; a wrong numeric answer). 100% shows the judge is not fooled by those, not that it separates two plausible answers; the real candidate-vs-base run is the harder test (position bias 0.18 there).
* Real run, candidate 20261010-033411 vs base (no adapter), 40 held-out items, 80 calls, 165,652 in + 79,165 out tokens, $0.56: win rate 0.438, cluster-bootstrap CI [0.325, 0.561], W/T/L 10/15/15, sign test p(better) 0.885. Deterministic battery (20 per category): math 7/20 vs 10/20 (REGRESSED), physics 7 vs 8, code 19 vs 18, research 10 vs 9. Held-out NLL still improves (2.774 -> 2.408). Gate: KEEP PRODUCTION (no significant pairwise win).

## Front-end (feature 1, LLM-FE): DEFAULT OFF everywhere
Policy `src/frontend.rs::decide_reason` (Rust owns the decision; `py_decide` is only its mirror for the tuning grid). Order: disabled -> forward; cache -> cache; tool -> tool;
OOD (max_sim < min_sim while min_sim > 0, or unknown) -> forward `ood`; no finite confidence or empty answer -> forward `no_conf`; conf < min_conf -> forward `low_conf`;
answer > max_local_tokens -> forward `too_long`; else local. `neural-mesh frontend` prints schema 2 `{ok, schema, enabled, decision, reason, tool, context, context_tokens}`.
* Fail closed: `rust_decide` (10 s timeout, exit code, `ok`, release binary not older than `src/frontend.rs` / `src/main.rs`) and every consumer forward the
  ORIGINAL prompt with NO context on any error; missing logprobs -> conf None (never a -99 sentinel).
* Verified cache only (`verified_cache_hit`): exact normalised match against STRICTLY EARLIER rows of the SAME project, outcome success, judge_score >= 0.9 with a
  judge_model or rubric recorded, exactly one distinct answer for the key, age <= 24 h, and an unchanged git HEAD when the instruction touches code. Today: 0 repeated
  instructions in 203 rows, so it never hits.
* Tool signal: `python -m llm tool-exemplars` writes `models/tool_exemplars.jsonl` (ROUTE3, MiniLM-384); Rust `--tool-exemplars F --query-vec V` routes through
  `KnnRouter::route_guarded` (tau .6, smin .4, k 5) restricted to `--tool-allow calc,units,csp`; bad dim / NaN / low share / far query / other label -> no tool.
  There is no tool executor: in `ask --mode cli` a tool decision forwards with context (R88: ~0 token value on agent work).
* Context: Rust picks ids (score order, <= 5 chunks, <= 800 tokens); `assemble_context` maps them to the exact counted text (first 4000 chars), drops unknown ids,
  drops EVERYTHING if over budget, and wraps each chunk as `<retrieved untrusted id=.. score=..>..</retrieved>` (closing tags inside the text are neutralised).

Consumers (all opt-in):
* S1 shadow (proposed, NOT installed): `hooks/settings-frontend-shadow.json` = UserPromptSubmit `cd C:/Users/mariu/neural-mesh && python -S -m llm hook --shadow`,
  timeout 5 s. It appends `{ts, sha256, session, project, prompt}` to `models/frontend_shadow.jsonl`, prints nothing, always exits 0 (never 2), off switch
  `NEURAL_MESH_FE_SHADOW=off`, 1 s watchdog, measured ~58 ms wall per prompt (bare interpreter ~36 ms). The expensive part is offline:
  `python -m llm shadow-drain` (takes the llm lock): signals, local answer, Rust decision, join to Claude's real answer (same session, same normalised text,
  from the ingestor's sft rows), judge -> `models/frontend_shadow_drained.jsonl` (no text). Activate only after review: merge the `UserPromptSubmit` entry into
  the repo's `.claude/settings.json` NEXT TO the recall hook (both run; recall prints memories, this prints nothing). Deactivate: remove it, or set the env switch.
* S2 `python -m llm ask [--enabled] [--mode shadow|advisory|cli] [--prompt-file F | stdin] [--forward-cmd CMD] [--local-timeout 20] [--log-text]`: pure pass-through
  without `--enabled`; shadow = forward unchanged and log `would_be`; advisory = forward with untrusted context; cli = serve cache/local answers. Skips the local
  model while `models/nightly.lock` is fresh (< 12 h). `--forward-cmd` gets the payload on stdin; its exit code is returned. Log `models/frontend_live.jsonl`
  (sha256 only unless `--log-text`). Dispatched before the llm lock.
* S3 (context-only UserPromptSubmit hook) NOT built: it would duplicate `hooks/recall_hook.py` (same event, same stdout-context channel, same purpose: inject
  retrieved snippets from earlier sessions). If wanted, add the sft-answer corpus as a second pool of the recall hook instead of a second injecting hook.

Evaluation `python -m llm frontend-eval` (llm/frontend_eval.py, `models/frontend_eval.json` schema 2): units = sft_meta clusters (fallback: session); trained rows
and their whole clusters dropped; units in first-ts order cut into contiguous thirds TUNE / HELD / CONF; rolling-origin bank (ts < row.ts); worlds = `source`.
Gates per split: n >= 30, sessions >= 3, clusters >= 3, judge decoy-reject >= .70, worlds >= 2, else INSUFFICIENT (structural gates checked before any model call).
Grid on TUNE only: min-conf {-1, -.7, -.5, -.3, -.2, -.15, -.1, -.05, -.02} x min-sim {.3 .. .7}; best = max saved tokens with wrong rate <= .10 and >= 10 local
answers, none -> REJECTED. HELD and CONF through the Rust binary; one Holm family over (split, world) x {saved > 0, quality loss < 0, TOST +-.01} (cluster
bootstrap, 4000 resamples). KEEP iff saved tokens Holm-significant in the same world on HELD and CONF, no Holm-significant loss, Wilson 95% upper bound of the
wrong rate <= .10 on HELD and CONF, and no Rust call failed. Only KEEP writes `models/frontend_cfg.json` (thresholds only; `ask` reads it only when
`verdict == KEEP`, and enabling is still the explicit `--enabled`). Today's data (203 rows, 1 session, 1 world, judge decoy-reject .60) gives INSUFFICIENT by
construction; v0 is logged REJECTED in RD-LOG (LLM-FE1). Do not run it before the release binary is rebuilt (`cargo build --release`; enforced).

Promotion ladder (config constants `RUNGS`, `LADDER`, `DEMOTION` in llm/frontend.py; NOT an active canary, every rung change is a manual owner decision):
shadow -> canary (hash-selected 5% of requests, every canary action audited against Claude) -> on (10% still audited).
* R1 local answer: shadow -> canary needs >= 300 shadow rows, >= 30 clusters, 2 worlds, >= 14 days, wrong UCB95 <= .10 on HELD and CONF, saved tokens Holm-significant,
  TOST +-.01; canary -> on needs >= 100 audited with wrong UCB95 <= .05. Prerequisites: a human-prompt world, a LoRA trained on ~1000+ rows, calibrated
  confidence, judge decoy-reject >= .70.
* R2 verified cache: >= 50 verified hits, hit rate >= 2%, stale-wrong UCB95 <= .05; on: >= 100 hits, UCB95 <= .02 (blocked today: 0 repeats).
* R3 tool auto-exec (route_guarded + Identity::check_strict): >= 100 routed human rows, UCB95 <= .02; then 200 rows, UCB95 <= .01 (~0 value per R88).
* R4 local draft + Claude edits: paired replay >= 200 rows (costs API budget: owner approval), TOST quality, Holm-significant token/latency drop, anchoring UCB95 <= .05.
* R5 per-skill R1: per skill >= 30 rows and >= 3 clusters per split, Holm across skills, canary when UCB95 <= .05.
Kill switches / demotion: rolling 50 audited actions with wrong UCB95 above the rung's ceiling -> down one rung; weekly decoy-reject < .70, a production model
pointer change, a stale binary or a new world -> back to shadow; severity 1 (destructive command, bad tool execution, override rate > 2x predicted) -> off.
