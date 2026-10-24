"""Parse the protocol's closing ```json block out of an agent's free-text output, tolerating the usual damage.

parse_output(text) -> normalized dict or None. Never raises. Picks the LAST block that decodes to a dict with an
"answer" key (the restated task spec at the top of a reply is also JSON), else the last dict block, else the last
balanced {...} span that mentions "answer". Repairs: smart quotes, trailing commas, // comments, Python literals.
"""
from __future__ import annotations

import json
import re

KINDS = ("fact", "hypothesis", "result", "question", "plan", "failure", "method")
_FENCE = re.compile(r"```[ \t]*(json|JSON|json5)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
MAX_TEXT = 200_000


def _repair(s: str) -> str:
    s = s.strip().lstrip("\ufeff")
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"(?m)^\s*//.*$", "", s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    return s


def _loads(s: str):
    for cand in (s, _repair(s)):
        try:
            return json.loads(cand)
        except (ValueError, RecursionError):
            continue
    try:                                    # single-quoted pseudo-JSON
        import ast
        v = ast.literal_eval(_repair(s).replace("true", "True").replace("false", "False").replace("null", "None"))
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _balanced_spans(text: str):
    """All top-level {...} spans; braces inside JSON strings are ignored."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = depth > 0
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def find_block(text) -> dict | None:
    """The raw dict from the agent's output (see module doc), or None."""
    if not isinstance(text, str) or not text:
        return None
    text = text[-MAX_TEXT:]
    cands = []
    for lang, body in _FENCE.findall(text):
        b = body.strip()
        if lang or b.startswith("{"):
            v = _loads(b)
            if isinstance(v, dict):
                cands.append(v)
    if not any("answer" in c for c in cands):
        for span in _balanced_spans(text):
            if '"answer"' in span or "'answer'" in span:
                v = _loads(span)
                if isinstance(v, dict):
                    cands.append(v)
    with_answer = [c for c in cands if "answer" in c]
    return (with_answer or cands or [None])[-1]


def _conf(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return {"high": 0.8, "medium": 0.5, "low": 0.2}.get(v.strip().lower())
        pct = "%" in v
        v = float(m.group(0)) / (100.0 if pct else 1.0)
    if isinstance(v, (int, float)):
        v = float(v)
        if v != v:
            return None
        if 1.0 < v <= 100.0:
            v /= 100.0
        return min(1.0, max(0.0, v))
    return None


def _strs(v, limit=20) -> list[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    out = []
    for x in v[:limit]:
        if isinstance(x, (dict, list)):
            x = json.dumps(x, ensure_ascii=False)
        if x is not None and str(x).strip():
            out.append(str(x).strip())
    return out


def _true(v) -> bool:
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def _needs(v) -> tuple[bool, str]:
    """needs_human as true/"true" or {value: bool, reason}; a free-text reason string alone also counts as true."""
    if isinstance(v, dict):
        return _true(v.get("value")) or _true(v.get("needed")), str(v.get("reason") or "").strip()
    if isinstance(v, str) and v.strip() and v.strip().lower() not in ("true", "false", "no", "none", "null"):
        return True, v.strip()
    return _true(v), ""


def normalize(d) -> dict | None:
    """Coerce a raw block into the protocol schema; unknown keys are dropped, bad values become defaults."""
    if not isinstance(d, dict):
        return None
    hyps = []
    raw_hyps = d.get("hypotheses")
    for h in (raw_hyps if isinstance(raw_hyps, list) else [])[:10]:
        if isinstance(h, dict) and str(h.get("claim") or "").strip():
            hyps.append({"claim": str(h["claim"]).strip(), "test_plan": str(h.get("test_plan") or "").strip(),
                         "alternatives": _strs(h.get("alternatives"), 5)})
        elif isinstance(h, str) and h.strip():
            hyps.append({"claim": h.strip(), "test_plan": "", "alternatives": []})
    posts = []
    raw_posts = d.get("workspace_posts")
    for p in (raw_posts if isinstance(raw_posts, list) else [])[:20]:
        if not isinstance(p, dict):
            continue
        kind = str(p.get("kind") or "").strip().lower()
        content = p.get("content")
        if kind not in KINDS or content is None or not str(content).strip():
            continue
        posts.append({"kind": kind, "content": str(content).strip() if not isinstance(content, (dict, list))
                      else json.dumps(content, ensure_ascii=False), "verified": _true(p.get("verified")),
                      "evidence": _strs(p.get("evidence") if p.get("evidence") is not None else p.get("sources"), 10)})
    return {
        "answer": d.get("answer"),
        "confidence": _conf(d.get("confidence")),
        "evidence": _strs(d.get("evidence")),
        "assumptions": _strs(d.get("assumptions")),
        "hypotheses": hyps,
        "open_questions": _strs(d.get("open_questions")),
        "workspace_posts": posts,
        "needs_human": _needs(d.get("needs_human"))[0],
        "needs_human_reason": _needs(d.get("needs_human"))[1] or str(d.get("needs_human_reason") or "").strip(),
        "consequences": _strs(d.get("consequences")),
        "verified": _true(d.get("verified")),
        "failures": _strs(d.get("failures") if d.get("failures") is not None else d.get("tried")),
    }


def parse_output(text) -> dict | None:
    try:
        return normalize(find_block(text))
    except Exception:
        return None


def answer_text(v) -> str:
    """The JSON "answer" value as the string an ANSWER: line would carry ("" when it is unusable)."""
    if v is None or isinstance(v, (bool, dict)):
        return ""
    if isinstance(v, list):
        if v and all(isinstance(x, str) and len(x.strip()) == 1 for x in v):
            return "".join(x.strip() for x in v)          # ["A", "C"] -> "AC"
        return ", ".join(str(x) for x in v)
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v))
    return str(v).strip()