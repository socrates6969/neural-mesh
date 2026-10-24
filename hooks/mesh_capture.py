#!/usr/bin/env python
"""Claude Code PostToolUse hook (matcher "Task|Agent"): capture each subagent's result into the Global Workspace.

Parses the protocol's closing ```json block (agentkit/parse.py) and posts to train.memory.workspace with
task_id = session_id and agent = subagent_type:
  answer -> "result"; hypotheses -> "hypothesis"; open_questions -> "question"; failures -> "failure";
  needs_human (reason + consequences) -> "failure" when attempts failed, else "question" (an approval request);
  workspace_posts -> their own kind.
Every post is stored UNVERIFIED: "verified" in model output is never trusted (it is kept only as claimed_verified, for
display). verified=True is set only out of band: grader-verified training results, or a human via agentkit.approve.
No JSON block -> one low-confidence "result" with the first 400 characters. Every text passes ingest.scrub first;
anything with a SECRET-class hit is dropped, not posted.
It prints nothing and never changes the tool result. Always exit 0; budget NEURAL_MESH_CAPTURE_BUDGET_S (default 2 s).
Opt-out: NEURAL_MESH_CAPTURE=off (or NEURAL_MESH_WORKSPACE=off).
"""
import os
import sys

if os.environ.get("NEURAL_MESH_CAPTURE", "").strip().lower() == "off":
    sys.exit(0)

import json
import threading
import time

T0 = time.monotonic()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from agentkit import parse  # noqa: E402

MAX_POSTS = 12
MAX_CONTENT = 1500
FALLBACK_CHARS = 400


def result_text(resp) -> str:
    """Text of a Task/Agent tool_response in any of its shapes (str, {content: [{type: text, text}]}, {result}, ...)."""
    out = []

    def walk(v, depth=0):
        if depth > 6 or len(out) > 200:
            return
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, dict):
            if isinstance(v.get("text"), str):
                out.append(v["text"])
            for k in ("content", "result", "output", "message", "response"):
                if k in v:
                    walk(v[k], depth + 1)
    walk(resp)
    return "\n".join(out)


def _scrub(text):
    """(clean text, ok). ok=False when the text holds a secret: such items are never posted."""
    try:
        from ingest.scrub import scrub
    except Exception:
        return "", False                       # no scrubber, no posting: if in doubt, drop
    r = scrub(str(text)[:MAX_CONTENT * 2])
    return r.text[:MAX_CONTENT], not r.secret


def plan_posts(d, text: str) -> list[dict]:
    """The workspace posts for one subagent result: [{kind, content, confidence, claimed, sources}]."""
    if d is None:
        snippet = " ".join((text or "").split())[:FALLBACK_CHARS]
        return [{"kind": "result", "content": snippet, "confidence": 0.2, "sources": []}] if snippet else []
    conf = d["confidence"] if d["confidence"] is not None else 0.5
    ev = d["evidence"]
    posts = []
    ans = parse.answer_text(d["answer"]) if not isinstance(d["answer"], dict) else json.dumps(d["answer"], ensure_ascii=False)
    if ans:
        posts.append({"kind": "result", "content": ans, "confidence": conf, "claimed": d["verified"], "sources": ev})
    for p in d["workspace_posts"]:
        cites = p["evidence"] or ev
        posts.append({"kind": p["kind"], "content": p["content"], "confidence": conf, "claimed": p["verified"],
                      "sources": cites})
    for h in d["hypotheses"]:
        c = h["claim"]
        if h["test_plan"]:
            c += " | test: " + h["test_plan"]
        if h["alternatives"]:
            c += " | alternatives: " + "; ".join(h["alternatives"])
        posts.append({"kind": "hypothesis", "content": c, "confidence": min(conf, 0.5), "sources": []})
    for q in d["open_questions"]:
        posts.append({"kind": "question", "content": q, "confidence": 0.5, "sources": []})
    for fl in d["failures"]:
        posts.append({"kind": "failure", "content": fl, "confidence": 0.5, "sources": []})
    if d["needs_human"]:
        c = "needs_human: " + (d["needs_human_reason"] or ans or "agent could not finish")
        if d["consequences"]:
            c += " | consequences: " + "; ".join(d["consequences"])
        if d["failures"]:
            c += " | tried: " + "; ".join(d["failures"])
        posts.append({"kind": "failure" if d["failures"] or not d["consequences"] else "question", "content": c,
                      "confidence": 0.5, "sources": []})
    return posts[:MAX_POSTS]


def capture(data, ws, deadline=None) -> list:
    """Post one PostToolUse payload's result to ws; returns the ids (or None per failed post). Never raises."""
    try:
        if data.get("tool_name") not in ("Task", "Agent"):
            return []
        ti = data.get("tool_input") or {}
        agent = str((ti.get("subagent_type") if isinstance(ti, dict) else "") or "general-purpose")
        task_id = str(data.get("session_id") or "")
        if not task_id:
            return []
        text = result_text(data.get("tool_response"))
        ids = []
        for p in plan_posts(parse.parse_output(text), text):
            if deadline is not None and time.monotonic() > deadline:
                break
            content, ok = _scrub(p["content"])
            if not ok or not content.strip():
                continue
            sources = []
            for s in p["sources"][:10]:
                sc, sok = _scrub(s)
                if sok and sc.strip():
                    sources.append(sc[:300])
            ids.append(ws.post(task_id, agent, p["kind"], content, confidence=float(p["confidence"]),
                               sources=sources or None, field=None, claimed_verified=bool(p.get("claimed"))))
        return ids
    except Exception:
        return []


def main():
    try:
        budget = float(os.environ.get("NEURAL_MESH_CAPTURE_BUDGET_S", 2.0))
    except ValueError:
        budget = 2.0

    def work():
        try:
            raw = sys.stdin.read()
            data = json.loads(raw) if raw.strip() else {}
            from agentkit import workspace
            if workspace.backend() is not None:
                capture(data, workspace, deadline=T0 + budget - 0.2)
        except Exception:
            pass

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(max(0.0, budget - (time.monotonic() - T0)))
    os._exit(0)


if __name__ == "__main__":
    main()