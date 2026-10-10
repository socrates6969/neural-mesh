"""LLM judge (approved feature 3): grade an agent output 0..1 with the LOCAL model through the Ollama HTTP API.

Deterministic-ish: temperature 0, seed 0, fixed num_ctx / num_predict, JSON output. Cached by a content hash of
(model, rubric version, prompt) in judge.jsonl, so a re-run never calls the model twice for the same input.
Only localhost is ever contacted. Inputs are already scrubbed.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import structural
from .scrub import scrub_text
from .util import append_jsonl, iter_jsonl, sha16

RUBRIC_VERSION = "judge-v1"
MAX_TASK = 2500
MAX_OUT_HEAD = 1500
MAX_OUT_TAIL = 2200

RUBRIC = """You are a strict reviewer of an AI coding agent's work. You see the TASK it was given and its FINAL OUTPUT.
Grade three parts and return ONLY JSON:
{"met_task": 0..0.5, "verified": 0..0.3, "clean": 0..0.2, "reason": "<= 20 words"}
- met_task: 0.5 = the output shows the task was fully done as asked; 0.25 = partly; 0 = not done, refused, or off-task.
- verified: 0.3 = concrete evidence (tests/commands run with results, file:line references, measured numbers); 0.15 = some; 0 = bare claims.
- clean: 0.2 = no unresolved blockers, errors or follow-up fix rounds needed; 0.1 = minor open issues; 0 = blocked, failing, or needs a redo.
Do not reward length or confidence. Unverified claims of success get low "verified"."""


def _clip(text: str, head: int, tail: int) -> str:
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n[...]\n" + text[-tail:]


def build_prompt(task: str, output: str) -> str:
    return f"{RUBRIC}\n\nTASK:\n{_clip(task, MAX_TASK, 0)}\n\nFINAL OUTPUT:\n{_clip(output, MAX_OUT_HEAD, MAX_OUT_TAIL)}\n\nJSON:"


def parse_grade(raw: str) -> tuple[float, str] | None:
    try:
        o = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        s, e = raw.find("{"), raw.rfind("}")
        if s < 0 or e <= s:
            return None
        try:
            o = json.loads(raw[s : e + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(o, dict):
        return None

    def num(k, hi):
        try:
            return max(0.0, min(hi, float(o.get(k, 0) or 0)))
        except (TypeError, ValueError):
            return 0.0

    if "score" in o and not any(k in o for k in ("met_task", "verified", "clean")):
        score = num("score", 1.0)
    else:
        score = num("met_task", 0.5) + num("verified", 0.3) + num("clean", 0.2)
    return round(min(1.0, score), 4), scrub_text(str(o.get("reason", ""))[:160])


class Judge:
    def __init__(self, cache_path: Path, url: str, model: str, budget: int, timeout_s: float = 120.0):
        self.cache_path = cache_path
        self.url = url.rstrip("/")
        self.model = model
        self.budget = budget
        self.timeout_s = timeout_s
        self.cache: dict[str, dict] = {}
        for r in iter_jsonl(cache_path):
            if r.get("id"):
                self.cache[r["id"]] = r
        self.stats = {"called": 0, "cached": 0, "failed": 0, "skipped_budget": 0, "unavailable": 0, "prefiltered": 0}
        self._available: bool | None = None

    def key(self, prompt: str) -> str:
        return "judge:" + sha16(self.model, RUBRIC_VERSION, prompt)

    def available(self) -> bool:
        if self._available is None:
            try:
                with urllib.request.urlopen(self.url + "/api/tags", timeout=5) as r:
                    tags = json.loads(r.read().decode())
                self._available = any(m.get("name") == self.model or m.get("model") == self.model for m in tags.get("models", []))
            except (urllib.error.URLError, OSError, ValueError):
                self._available = False
        return self._available

    def cached(self, task: str, output: str) -> dict | None:
        return self.cache.get(self.key(build_prompt(task, output)))

    def grade(self, target: str, task: str, output: str) -> dict | None:
        """Return the cached or fresh grade {score, reason, ...}, or None (no budget / model unavailable / failure)."""
        bad = structural.check(output)
        if bad:   # $0 structural pre-check: obvious junk (empty / refusal / truncated / looping) never reaches the model
            self.stats["prefiltered"] += 1
            return {"id": "structural:" + bad, "target": target, "score": 0.0, "reason": "structural:" + bad, "model": "structural",
                    "rubric_version": RUBRIC_VERSION, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        prompt = build_prompt(task, output)
        k = self.key(prompt)
        if k in self.cache:
            self.stats["cached"] += 1
            return self.cache[k]
        if self.budget <= 0:
            self.stats["skipped_budget"] += 1
            return None
        if not self.available():
            self.stats["unavailable"] += 1
            return None
        self.budget -= 1
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False, "format": "json",
                           "options": {"temperature": 0, "seed": 0, "top_k": 1, "num_ctx": 8192, "num_predict": 160}}).encode()
        req = urllib.request.Request(self.url + "/api/generate", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                resp = json.loads(r.read().decode())
        except (urllib.error.URLError, OSError, ValueError):
            self.stats["failed"] += 1
            return None
        g = parse_grade(resp.get("response", ""))
        if g is None:
            self.stats["failed"] += 1
            return None
        self.stats["called"] += 1
        rec = {"id": k, "target": target, "score": g[0], "reason": g[1], "model": self.model, "rubric_version": RUBRIC_VERSION,
               "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        append_jsonl(self.cache_path, [rec])
        self.cache[k] = rec
        return rec
