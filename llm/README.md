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
  `llm/frontend.py`: exact cache, MiniLM max-cosine OOD guard, Ollama `logprobs` confidence, answer length.
* Nightly (provided, not installed): `llm/scripts/run_nightly.ps1` (runs the ingestor first, then train/eval/promote).

## Pairwise judge: measured 2026-10-10 (Claude Haiku via `claude -p --model haiku` = claude-haiku-4-5; every number from `models/pair_calibration.json` / `adapters/20261010-033411/eval.json`)
* Calibration PASSED (148 calls, 333,976 input + 97,064 output tokens, $0.82): decoy test 44/44 (real answer preferred in both orders; n = 44 pairs from 203 SFT rows, decoy = answer to the MiniLM-nearest different request with response cosine < 0.85); deterministic agreement 30/30 (11 math + 16 physics + 3 code bench pairs: harness-correct Claude answer vs harness-wrong local answer, both orders). Position bias (same letter in both orders) 0.0 in calibration.
  Caveat: both sets are easy by construction (a wrong-topic answer; a wrong numeric answer). 100% shows the judge is not fooled by those, not that it separates two plausible answers; the real candidate-vs-base run is the harder test (position bias 0.18 there).
* Real run, candidate 20261010-033411 vs base (no adapter), 40 held-out items, 80 calls, 165,652 in + 79,165 out tokens, $0.56: win rate 0.438, cluster-bootstrap CI [0.325, 0.561], W/T/L 10/15/15, sign test p(better) 0.885. Deterministic battery (20 per category): math 7/20 vs 10/20 (REGRESSED), physics 7 vs 8, code 19 vs 18, research 10 vs 9. Held-out NLL still improves (2.774 -> 2.408). Gate: KEEP PRODUCTION (no significant pairwise win).
