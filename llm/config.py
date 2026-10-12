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
