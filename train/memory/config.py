"""Paths and tuned constants for the shared method-card memory (all live data is under data/lifelong/, gitignored)."""
from __future__ import annotations

import os
from pathlib import Path

NM = Path(os.environ.get("NEURAL_MESH_ROOT", str(Path.home() / "neural-mesh")))
MEM_DIR = Path(os.environ.get("NEURAL_MESH_MEMORY", str(NM / "data" / "lifelong" / "mesh" / "memory")))
LABELS = Path(os.environ.get("NEURAL_MESH_MEMORY_LABELS", str(NM / "data" / "lifelong" / "mesh" / "memory" / "labels")))              # offline eval label sources (Omni-MATH domains, Codeforces tags, LLM topics)
RUNS = Path(os.environ.get("MESHTRAIN_ROOT", r"C:\Users\mariu\ruv\hive\runs\agent-catalog"))

MINILM = Path(os.environ.get("MESH_MINILM", str(Path.home() / ".ruvector" / "models" / "all-MiniLM-L6-v2")))
CROSS = Path(os.environ.get("MESH_CROSS_ENCODER", str(Path.home() / ".ruvector" / "models" / "ms-marco-MiniLM-L6-v2")))
OLLAMA = os.environ.get("MESH_OLLAMA", "http://127.0.0.1:11434")
DISTILL_MODEL = os.environ.get("MESH_DISTILL_MODEL", "qwen2.5-coder:7b")
# extractive (deterministic, no model) is the default; "auto" tries Ollama first. Only use auto when the GPU is free.
DISTILL = os.environ.get("MESH_CARD_DISTILL", "extractive")

KEY = Path(os.environ.get("NEURAL_MESH_MEMORY_KEY", str(Path.home() / ".claude" / "mesh-memory.key")))
PORT = int(os.environ.get("NEURAL_MESH_MEMORY_PORT", "47391"))
IDLE_EXIT_S = float(os.environ.get("NEURAL_MESH_MEMORY_IDLE_S", 20 * 60))

CARD_TOKENS = 120        # max tokens of one card's injected text
INJECT_TOKENS = 350      # max tokens of the whole injected block
MAX_PROBLEM = 1500       # chars of problem text kept for matching (never injected)
RRF_K = 60
RERANK_TOP = 20


LIVE_MEM_DIR = NM / "data" / "lifelong" / "mesh" / "memory"
LIVE_WORKSPACE_DB = NM / "data" / "lifelong" / "mesh" / "workspace.db"


def under_test() -> bool:
    """True inside any test run: MESH_TEST=1 (set by importing train/memory/tests, and inherited by subprocesses),
    a pytest run (PYTEST_CURRENT_TEST, or pytest loaded), or a unittest/pytest command line. The last two catch
    runners that never set PYTEST_CURRENT_TEST, e.g. `python -m unittest`."""
    import sys
    if os.environ.get("MESH_TEST", "") == "1" or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    if "_pytest" in sys.modules:
        return True
    argv0 = " ".join(sys.argv[:1]).replace("\\", "/").lower()
    return "unittest" in argv0 or "pytest" in argv0


def guard_live(path) -> None:
    """Tests must never write the live memory: raise if running under pytest / MESH_TEST=1 and `path` is live data.
    Point tests at temp files with MESH_WORKSPACE_DB (workspace) and NEURAL_MESH_MEMORY (card store)."""
    if not under_test():
        return
    p = Path(path).resolve()
    for live in (LIVE_WORKSPACE_DB.resolve(), LIVE_MEM_DIR.resolve()):
        if p == live or live in p.parents:
            raise RuntimeError("refusing to touch live mesh memory under test: %s (set MESH_WORKSPACE_DB / "
                               "NEURAL_MESH_MEMORY to a temp path)" % p)


def cards_path() -> Path:
    return MEM_DIR / "cards.jsonl"


def stats_path() -> Path:
    return MEM_DIR / "usefulness.json"


def injections_path() -> Path:
    return MEM_DIR / "injections.jsonl"


# Splits whose items are never returned (comma list, e.g. "held"): protects held-out evaluation done through live
# sessions. Same-problem items are always excluded (see problem_key), whatever this says.
EXCLUDE_SPLITS = frozenset(s.strip() for s in os.environ.get("MESH_MEMORY_EXCLUDE_SPLITS", "").split(",") if s.strip())
PROBLEM_KEY_CHARS = 160


def norm_text(text: str) -> str:
    """Lowercase alphanumerics only - robust to whitespace, LaTeX spacing, punctuation and scrub placeholders."""
    import re
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def problem_key(problem: str) -> str:
    """Fingerprint of a solved problem: its first PROBLEM_KEY_CHARS normalised characters ("" if too short to be
    distinctive). A query that CONTAINS this key is the same problem, so the item would leak its own answer."""
    k = norm_text(problem)[:PROBLEM_KEY_CHARS]
    return k if len(k) >= 40 else ""


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars per token for English/LaTeX), deliberately a little pessimistic."""
    return (len(text) + 3) // 4
