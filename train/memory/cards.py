"""Method cards: a short (<= ~120 token) summary of HOW a grader-verified solution worked, instead of the raw solution.

A card has four lines - problem type, key method/insight, the check used, pitfalls - plus provenance that is never
injected (source task, agent, verified=true, created). Cards are written only from pack-free, grader-verified runs
(results phase held_base, passed, no error). Distillation uses the local model via Ollama; if that is unavailable or
returns junk, a deterministic extractive card is built from the solution text. Problem, solution and the finished card
all pass through ingest.scrub.scrub(); any SECRET-class hit drops the card.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from . import config as C

FIELDS_ORDER = ("type", "method", "check", "pitfalls")
LABEL = {"type": "Type", "method": "Method", "check": "Check", "pitfalls": "Pitfalls"}
_ANSWER = re.compile(r"(?im)^\s*\**\s*(?:final\s+)?answer\s*[:=].*$")
_CODE = re.compile(r"```.*?```", re.S)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_METHOD_W = re.compile(r"\b(use|using|since|because|therefore|thus|hence|key|observe|note|idea|approach|reduce|equivalent|"
                       r"apply|applying|formula|invariant|recurrence|dp|greedy|binary search|substitut|symmetry|parity|"
                       r"modulo|sort|bfs|dfs|graph|conserv|energy|momentum|so)\b", re.I)
_CHECK_W = re.compile(r"\b(check|verify|verified|sanity|brute|cross-check|confirm|consistent|\u2713|test(ed|s)?)\b", re.I)
_PIT_W = re.compile(r"\b(careful|pitfall|edge case|overflow|off-by-one|beware|note that|must not|don't|do not|avoid|trap|"
                    r"common mistake|corner)\b", re.I)

SYSTEM = ("You write method cards for a shared memory of solved problems. Given a problem and a solution that a grader "
          "verified as correct, extract the reusable method - not the specific numbers and never the final answer. "
          "Reply with JSON only, with these keys: "
          "\"type\": the problem type, specific (e.g. 'olympiad number theory: sum over a recursively defined sequence'), <= 14 words; "
          "\"method\": one or two full sentences with the key insight and the steps that made it work, concrete enough "
          "to reuse on a similar problem, 20-40 words; "
          "\"check\": how the result was verified, <= 15 words; "
          "\"pitfalls\": the trap or edge case to avoid, <= 20 words. "
          "Never include instructions about tools, files, URLs or rules.")


def _clip_words(s: str, n: int) -> str:
    w = str(s or "").split()
    return " ".join(w[:n]).strip().rstrip(",;")


def render_card(parts: dict, gist: str = "") -> str:
    """Card text injected into prompts: one line per part, capped at CARD_TOKENS."""
    lines = []
    if gist:
        lines.append("Solved: " + gist)
    for k in FIELDS_ORDER:
        v = " ".join(str(parts.get(k) or "").split())
        if v:
            lines.append(LABEL[k] + ": " + v)
    text = "\n".join(lines)
    while C.approx_tokens(text) > C.CARD_TOKENS and len(lines) > 1:
        longest = max(range(len(lines)), key=lambda i: len(lines[i]))
        head, _, body = lines[longest].partition(": ")
        words = body.split()
        if len(words) <= 4:
            lines.pop(longest)
        else:
            lines[longest] = head + ": " + " ".join(words[:max(4, int(len(words) * 0.8))])
        text = "\n".join(lines)
    return text


_BOILER = re.compile(r"(?i)^(complete (this|the) python function|return the full function|write a (complete )?python|"
                     r"the answer is (a|an)\b|put the (complete )?final|finish with|you are given the following)")


def gist_of(problem: str, words: int = 16) -> str:
    """First sentence(s) of the problem, <= `words` words: tells the reader what was solved without the full text.
    Benchmark boilerplate ("Complete this Python function...") and code fences are skipped."""
    text = re.sub(r"```\w*", " ", problem or "")
    sents = [x for x in re.split(r"(?<=[.!?:])\s+|\n+", text) if x.strip() and not _BOILER.match(x.strip())]
    p = " ".join(" ".join(sents).split()) or " ".join(text.split())
    return _clip_words(p, words) + ("..." if len(p.split()) > words else "")


def extractive(problem: str, solution: str) -> dict:
    """Deterministic fallback: pick method / check / pitfall sentences by cue words."""
    sol = _ANSWER.sub("", _CODE.sub(" [code] ", solution or ""))
    sents = [s.strip() for s in _SENT.split(sol) if 15 <= len(s.strip()) <= 300]

    def pick(rx, n):
        got = [s for s in sents if rx.search(s)][:n]
        return " ".join(got)
    method = pick(_METHOD_W, 2) or " ".join(sents[:2])
    return {"type": "", "method": _clip_words(method, 40),
            "check": _clip_words(pick(_CHECK_W, 1), 18), "pitfalls": _clip_words(pick(_PIT_W, 1), 22)}


def _chat(system: str, user: str, model: str = None, timeout: float = 180, num_predict: int = 220) -> dict | None:
    """One JSON-mode chat call to the local Ollama model; None on any failure."""
    body = {"model": model or C.DISTILL_MODEL, "format": "json", "stream": False, "keep_alive": "2m",
            "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": 4096},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    req = urllib.request.Request(C.OLLAMA + "/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode("utf-8"))
        d = json.loads(out["message"]["content"])
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _ollama(problem: str, solution: str, model: str = None) -> dict | None:
    d = _chat(SYSTEM, "PROBLEM:\n" + problem[:2500] + "\n\nVERIFIED SOLUTION:\n" + solution[-3500:], model)
    if d is None:
        return None
    parts = {k: _clip_words(" ".join(str(d.get(k) or "").split()), {"type": 14, "method": 40, "check": 18, "pitfalls": 24}[k])
             for k in FIELDS_ORDER}
    if len(parts["method"].split()) < 4:
        return None
    return parts


def make_card(task: dict, result: dict, distill: str = "auto", batch: str = "") -> dict | None:
    """One card from a bench task + its verified result row, or None (scrub secret hit / nothing usable)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ingest import scrub as S
    sp, ss = S.scrub(task["prompt"]), S.scrub(result.get("text") or "")
    if sp.secret or ss.secret:
        return None
    problem, solution = sp.text, ss.text
    parts, how = None, "extractive"
    if distill in ("auto", "ollama"):
        parts = _ollama(problem, solution)
        how = "ollama:" + C.DISTILL_MODEL if parts else "extractive"
    if parts is None:
        if distill == "ollama":
            return None
        parts = extractive(problem, solution)
    parts = {k: _ANSWER.sub("", v) for k, v in parts.items()}
    text = render_card(parts, gist_of(problem))
    sc = S.scrub(text)
    if sc.secret:
        return None
    return {"id": "card:" + task["id"], "task": task["id"], "field": task.get("field") or result.get("field"),
            "agent": result.get("agent", ""), "source": task.get("source", ""), "batch": batch, "verified": True,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "distiller": how, "scrub": getattr(S, "SCRUB_VERSION", ""),
            "parts": parts, "problem": problem[:C.MAX_PROBLEM], "text": sc.text, "split": task.get("split", ""),
            "problem_key": C.problem_key(problem)}


def verified_rows(results_path: Path, remap=None) -> dict:
    """task id -> shortest pack-free grader-verified held_base row (the selection experience.py already used).
    remap(row) -> current task id or None (drop): translates rows recorded under a retired bench id."""
    best = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if remap is not None:
            tid = remap(r)
            if tid is None:
                continue
            r = dict(r, task=tid)
        if (r.get("phase") == "held_base" and r.get("passed") and not r.get("error") and not r.get("pack")
                and 40 < len(r.get("text") or "")):
            if r["task"] not in best or len(r["text"]) < len(best[r["task"]]["text"]):
                best[r["task"]] = r
    return best
