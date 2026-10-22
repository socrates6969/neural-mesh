"""sft.jsonl reading, time-based held-out split, new/replay selection, and a LABELLED synthetic sample.

Schema (ingestor v1, BUS.md): {id, ts, project, session, source, agent, model, instruction, response, outcome,
judge_score, judge_model, rubric_version, scrub_version, chars}. Rows are already scrubbed by the ingestor.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import config


def read_jsonl(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a torn last line from a concurrent writer: skip it, it will be complete next run
                if isinstance(r, dict) and r.get("instruction") and r.get("response") and r.get("id"):
                    out.append(r)
    except FileNotFoundError:
        pass
    # de-duplicate by id (append-only file; first occurrence wins)
    seen, uniq = set(), []
    for r in out:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    return uniq


def load_meta(path: Path | None = None) -> dict[str, dict]:
    """ingestor sidecar sft_meta.jsonl {id, cluster, cluster_size, heldout, replay_weight} (MiniLM-384 clusters)."""
    out = {}
    try:
        with open(path or config.DATA / "sft_meta.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["id"]] = r
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return out


def split(rows: list[dict], meta: dict[str, dict] | None = None) -> tuple[list[dict], list[dict]]:
    """Prefer the ingestor's cluster-aware held-out flag (whole paraphrase clusters whose first row is in the latest 20%,
    so no paraphrase leaks into training) when it covers every row; else the session-time split below."""
    if meta and all(r["id"] in meta for r in rows):
        return [r for r in rows if not meta[r["id"]].get("heldout")], [r for r in rows if meta[r["id"]].get("heldout")]
    return split_by_session_time(rows)


def split_by_session_time(rows: list[dict], frac: float = config.HELDOUT_FRAC) -> tuple[list[dict], list[dict]]:
    """Held-out = the latest sessions (ordered by each session's LAST ts), ~frac of sessions, at least one session when there
    are >= 2. Train never contains a held-out session. Deterministic for a given file."""
    last: dict[str, str] = {}
    for r in rows:
        s = r.get("session") or r["id"]
        t = r.get("ts") or ""
        if t > last.get(s, ""):
            last[s] = t
        last.setdefault(s, t)
    sessions = sorted(last, key=lambda s: (last[s], s))
    if len(sessions) < 2:
        return rows, []
    n_hold = max(1, round(len(sessions) * frac))
    held = set(sessions[-n_hold:])
    train = [r for r in rows if (r.get("session") or r["id"]) not in held]
    test = [r for r in rows if (r.get("session") or r["id"]) in held]
    return train, test


def select_for_round(train: list[dict], trained_ids: set[str], replay_ratio: float = config.REPLAY_RATIO, seed: int = config.SEED,
                     weights: dict[str, float] | None = None) -> tuple[list[dict], list[dict]]:
    """New = train rows never trained on. Replay = sample (without replacement) of already-trained rows, replay_ratio x
    len(new); weighted by the ingestor's replay_weight (1/cluster_size: diversity-aware) when given, else uniform."""
    new = [r for r in train if r["id"] not in trained_ids]
    old = [r for r in train if r["id"] in trained_ids]
    k = min(len(old), int(round(len(new) * replay_ratio)))
    rng = random.Random(seed + len(trained_ids))
    if not k:
        return new, []
    if not weights:
        return new, rng.sample(old, k)
    # Efraimidis-Spirakis weighted sampling without replacement
    keyed = sorted(old, key=lambda r: -(rng.random() ** (1.0 / max(1e-6, float(weights.get(r["id"], 1.0))))))
    return new, keyed[:k]


# ------------------------------------------------------------------------------------------------
# Synthetic sample (LABELLED: every row has source "synthetic" and project "SYNTHETIC"). Only used until the
# ingestor's sft.jsonl exists, or for tests. Tasks are small code/ops Q&A with deterministic reference answers.

_SYN = [
    ("Write a Python function `is_even(n)` that returns True when n is even.", "def is_even(n):\n    return n % 2 == 0"),
    ("What does `git add -A` do and why should it be avoided in this repo?", "It stages every change in the working tree, including untracked files. Stage files by explicit path instead so nothing unintended is committed."),
    ("Convert 3 kilometres to metres.", "3 km = 3000 m."),
    ("Write a Rust function `clamp01(x: f32) -> f32`.", "fn clamp01(x: f32) -> f32 {\n    x.max(0.0).min(1.0)\n}"),
    ("How do you run only the library tests of a Rust crate in release mode?", "cargo test --release --lib"),
    ("Write a Python one-liner that reverses a list `xs`.", "xs[::-1]"),
    ("Name the PostgreSQL feature that enforces per-tenant row filtering.", "Row Level Security (RLS), enabled per table with policies; FORCE ROW LEVEL SECURITY also applies it to the table owner."),
    ("What is 17 * 23?", "391"),
    ("Write a JavaScript arrow function that squares a number.", "const square = (x) => x * x;"),
    ("How do you list the Ollama models installed locally?", "ollama list"),
    ("Write a Python function `mean(xs)` that returns 0.0 for an empty list.", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0"),
    ("What HTTP status code means 'Too Many Requests'?", "429"),
]


def synthetic_rows(n_sessions: int = 30, per_session: int = 6, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for s in range(n_sessions):
        for j in range(per_session):
            q, a = _SYN[(s * per_session + j) % len(_SYN)]
            variant = rng.choice(["", "Please answer briefly. ", "Task: "])
            ts = f"2026-{9 + s // 28:02d}-{1 + s % 28:02d}T{10 + j:02d}:00:00Z"
            rid = f"sft:syn{s:02d}{j:02d}"
            rows.append({"id": rid, "ts": ts, "project": "SYNTHETIC", "session": f"syn-session-{s:02d}", "source": "synthetic",
                         "agent": None, "model": None, "instruction": variant + q, "response": a, "outcome": "success",
                         "judge_score": 1.0, "judge_model": None, "rubric_version": "synthetic", "scrub_version": "synthetic",
                         "chars": {"instruction": len(variant + q), "response": len(a)}})
    return rows


def write_synthetic(path: Path) -> int:
    rows = synthetic_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def load_rows(synthetic: bool = False) -> tuple[list[dict], str]:
    """(rows, source label). Real sft.jsonl when it has rows, else the labelled synthetic sample when asked for."""
    if not synthetic:
        rows = read_jsonl(config.SFT)
        if rows:
            return rows, f"real:{config.SFT}"
        raise SystemExit(f"no rows in {config.SFT}; pass --synthetic to use the labelled synthetic sample")
    return synthetic_rows(), "SYNTHETIC (labelled sample, not real data)"


def sample_external(n: int, seed: int = config.SEED) -> list[dict]:
    """Up to n rows from data/lifelong/sft_external.jsonl (open-source datasets + local git history, built by
    `python -m llm external`). These are TRAIN-ONLY: they never enter the held-out split, so nightly eval stays
    comparable. Sampled per source in proportion to its size, a fresh sample each night (seed varies)."""
    import random as _r
    if n <= 0 or not config.EXTERNAL.exists():
        return []
    rows = read_jsonl(config.EXTERNAL)
    rng = _r.Random(seed)
    rng.shuffle(rows)
    return rows[:n]
