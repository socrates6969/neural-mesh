"""Paths and knobs for the local LLM pipeline (owner: llm-smith). Everything stays on this machine."""
from __future__ import annotations

import json
import os
from pathlib import Path

PKG = Path(__file__).resolve().parent            # neural-mesh/llm
REPO = PKG.parent                                # neural-mesh
DATA = Path(os.environ.get("MESH_LLM_DATA", REPO / "data" / "lifelong"))
SFT = DATA / "sft.jsonl"
OUTCOMES = DATA / "outcomes.jsonl"
EXTERNAL = DATA / "sft_external.jsonl"
EVAL_EXTERNAL = DATA / "sft_eval_external.jsonl"  # held-out open-source questions for the pairwise judge (never trained on)
SELF_VERIFIED = DATA / "sft_self_verified.jsonl"  # rejection-sampled answers that passed an automatic checker (llm/selftrain.py); TRAIN-ONLY
DPO_PAIRS = DATA / "dpo_pairs.jsonl"              # checker-verified preference pairs (llm/selftrain.py); TRAIN-ONLY
TRAIN_MINUTES = float(os.environ.get("MESH_LLM_TRAIN_MINUTES", "110"))   # nightly training wall-clock budget
NIGHTLY_EPOCHS = int(os.environ.get("MESH_LLM_EPOCHS", "3"))             # upper bound; the time budget usually stops first
EXTERNAL_PER_NIGHT = int(os.environ.get("MESH_LLM_EXTERNAL_PER_NIGHT", "6000"))
SELF_PER_NIGHT = int(os.environ.get("MESH_LLM_SELF_PER_NIGHT", "1500"))        # self_verified rows mixed into one round
# (B) rejection-sampling stage before train: GPU minutes taken OUT of TRAIN_MINUTES (0 = off, the default)
SELFTRAIN_MINUTES = float(os.environ.get("MESH_LLM_SELFTRAIN_MINUTES", "0"))
# (C) DPO after SFT: on when at least DPO_MIN_PAIRS eligible pairs exist; its minutes come OUT of TRAIN_MINUTES
DPO_MIN_PAIRS = int(os.environ.get("MESH_LLM_DPO_MIN_PAIRS", "64"))
DPO_MINUTES = float(os.environ.get("MESH_LLM_DPO_MINUTES", "15"))
DPO_JUDGE_PAIRS = os.environ.get("MESH_LLM_DPO_JUDGE", "1") != "0"          # pairs from pairwise-judge verdicts (train rows only)
DPO_SELF_PAIRS = os.environ.get("MESH_LLM_DPO_SELF", "1") != "0"            # pairs from checker verdicts (llm/selftrain.py)
# (D) continual lineage: train from the lineage head (best not-worse candidate) instead of from base
CONTINUAL = os.environ.get("MESH_LLM_CONTINUAL", "1") != "0"
MAX_CHAIN = int(os.environ.get("MESH_LLM_MAX_CHAIN", "7"))
# wall-clock end of the nightly llm round, "HH:MM" local (run_nightly.ps1 sets 07:30; unset = no deadline): llm/deadline.py
NIGHTLY_DEADLINE = os.environ.get("MESH_NIGHTLY_DEADLINE", "")
PROMOTE_RESERVE_MIN = float(os.environ.get("MESH_LLM_PROMOTE_RESERVE_MIN", "10"))   # kept free before the deadline for promote
JUDGE_RESERVE_MIN = float(os.environ.get("MESH_LLM_JUDGE_RESERVE_MIN", "30"))       # generation stops this long before it                   # unpromoted generations on top of production before a reset
MODELS = Path(os.environ.get("MESH_LLM_MODELS", PKG / "models"))
BASE_DIR = MODELS / "base"
ADAPTERS = MODELS / "adapters"
MERGED = MODELS / "merged"
STATE = MODELS / "state.json"                    # trained ids, production pointer, history
BASE_CHOICE = MODELS / "base_choice.json"        # written by `python -m llm bench`
JUDGE_CACHE = MODELS / "judge_cache.jsonl"       # our own cache (the ingestor owns data/lifelong/judge.jsonl)
LOCK = MODELS / "nightly.lock"
RUNS = MODELS / "runs.jsonl"                   # one RunRecord per `python -m llm nightly` (llm/nightly.py)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
JUDGE_MODEL = os.environ.get("MESH_LLM_JUDGE", "qwen2.5-coder:7b")
OLLAMA_NAME = "mesh-llm"

DEFAULT_BASE = "Qwen2.5-Coder-1.5B-Instruct"
HELDOUT_FRAC = 0.2          # latest sessions (by their last ts) held out, never trained on
REPLAY_RATIO = 1.0          # replay rows per new row (sampled from older trained rows)
ANCHOR_RATIO = 1.0          # self-distillation anchor rows per new row (llm/anchors.py; 0 disables)
MAX_LEN = 1024              # tokens per training example (prompt + response, truncated from the left of the prompt)
SEED = 1234


def base_name() -> str:
    if os.environ.get("MESH_LLM_BASE"):
        return os.environ["MESH_LLM_BASE"]
    try:
        return json.loads(BASE_CHOICE.read_text(encoding="utf-8"))["base"]
    except (OSError, KeyError, ValueError):
        return DEFAULT_BASE


def base_path(name: str | None = None) -> Path:
    return BASE_DIR / (name or base_name())


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"trained_ids": [], "production": None, "history": []}


def save_state(st: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
    os.replace(tmp, STATE)
