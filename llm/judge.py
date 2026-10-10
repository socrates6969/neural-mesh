"""Local judge via the Ollama HTTP API (temperature 0, seed 0, JSON output), cached by content hash. Nothing leaves the box."""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request

from . import config

RUBRIC_VERSION = "llm-smith-judge-v1"
RUBRIC = (
    "You grade a CANDIDATE answer against a REFERENCE answer written by a strong assistant for the same request.\n"
    "Score 1.0 if the candidate is correct and would satisfy the request as well as the reference (wording may differ);\n"
    "0.5 if partly correct or missing important parts; 0.0 if wrong, off-topic, refuses, or invents facts.\n"
    "Judge substance, not style or length. Reply ONLY with JSON: {\"score\": <0|0.5|1>, \"why\": \"<10 words>\"}."
)


def _post(path: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(config.OLLAMA_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _cache() -> dict:
    out = {}
    try:
        with open(config.JUDGE_CACHE, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r["key"]] = r
                except (ValueError, KeyError):
                    pass
    except FileNotFoundError:
        pass
    return out


_CACHE: dict | None = None


def grade(instruction: str, reference: str, candidate: str, model: str = config.JUDGE_MODEL, max_chars: int = 6000) -> float:
    global _CACHE
    if _CACHE is None:
        _CACHE = _cache()
    prompt = (f"{RUBRIC}\n\nREQUEST:\n{instruction[-max_chars:]}\n\nREFERENCE:\n{reference[:max_chars]}\n\n"
              f"CANDIDATE:\n{candidate[:max_chars]}\n")
    key = hashlib.sha256(f"{model}\x00{RUBRIC_VERSION}\x00{prompt}".encode()).hexdigest()[:24]
    if key in _CACHE:
        return _CACHE[key]["score"]
    r = _post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "format": "json",
                                "options": {"temperature": 0, "seed": 0, "num_ctx": 8192}})
    score = parse_score(r.get("response", ""))
    rec = {"key": key, "score": score, "model": model, "rubric": RUBRIC_VERSION}
    config.MODELS.mkdir(parents=True, exist_ok=True)
    with open(config.JUDGE_CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    _CACHE[key] = rec
    return score


def parse_score(text: str) -> float:
    """robust parse; anything unparseable counts as 0.0 (fail closed)."""
    try:
        v = float(json.loads(text).get("score"))
    except (ValueError, TypeError, AttributeError):
        m = re.search(r'"score"\s*:\s*([0-9.]+)', text or "")
        v = float(m.group(1)) if m else 0.0
    return min(1.0, max(0.0, v))


def unload(model: str = config.JUDGE_MODEL) -> None:
    """free the judge's VRAM before training (keep_alive 0)."""
    try:
        _post("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
    except OSError:
        pass


def ollama_generate(model: str, prompt: str, max_new: int = 256) -> tuple[str, int]:
    r = _post("/api/generate", {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "seed": 0, "num_predict": max_new}})
    return r.get("response", "").strip(), int(r.get("eval_count", 0))
