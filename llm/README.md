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
* Eval gate: held-out response-token NLL (cluster-bootstrap 95% CI must exclude 0) AND judge battery not significantly
  worse and mean >= -0.05 AND the judge must reject >= 70% of nearest-but-wrong (MiniLM-384) decoys AND n >= 30, >= 3 clusters.
* Promote: merge on CPU -> llama.cpp `convert_hf_to_gguf.py` (official release tarball in models/tools/) -> q8_0 GGUF ->
  `ollama create mesh-llm:<ver>` -> `ollama cp` to `mesh-llm:latest`. (Ollama 0.40's own safetensors import rejects Qwen2.)
* Front-end (feature 1): policy in `src/frontend.rs`, CLI `neural-mesh frontend` (default OFF). Signals from
  `llm/frontend.py`: exact cache, MiniLM max-cosine OOD guard, Ollama `logprobs` confidence, answer length.
* Nightly (provided, not installed): `llm/scripts/run_nightly.ps1` (runs the ingestor first, then train/eval/promote).
