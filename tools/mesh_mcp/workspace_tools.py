r"""Global Workspace tools: thin wrappers over train.memory.workspace (branch exp/shared-memory, by mesh-memory).

API wrapped exactly: post(task_id, agent, kind, content, confidence, verified, sources, field) -> int|None;
read(task_id, query, k) -> list[dict]; search(text, field, k) -> cards; analogies(text, k, field);
observe(ids, passed); stats() -> dict.
The module is imported from MESH_WORKSPACE_ROOT (default C:\Users\mariu\neural-mesh). Until it exists every tool
returns a clear "workspace not available" message instead of failing. Only a successful import is cached.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import sys

from tools.mesh_mcp.common import cap

DEFAULT_ROOT = r"C:\Users\mariu\neural-mesh"
CLAIM_TAG = "[agent claims verified; unchecked] "  # same tag as agentkit/workspace.py
KINDS = ("fact", "hypothesis", "result", "question", "plan", "failure", "method")
_mod = None


def _workspace():
    global _mod
    if _mod is not None:
        return _mod, None
    root = os.environ.get("MESH_WORKSPACE_ROOT", DEFAULT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        _mod = importlib.import_module("train.memory.workspace")
        return _mod, None
    except Exception as e:  # noqa: BLE001 - any import failure means "not available yet"
        return None, ("workspace not available: train.memory.workspace could not be imported from %s (%s: %s). "
                      "The Global Workspace is being built on branch exp/shared-memory; until it is merged, keep "
                      "notes in your own output." % (root, type(e).__name__, str(e)[:200]))


def _accepts(fn, name: str) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values())


def _task(task_id):
    return task_id or os.environ.get("CLAUDE_SESSION_ID") or "default"


def _list(value):
    if value is None or isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return list(value)


def _render(result, empty: str) -> str:
    if result is None or result == [] or result == {}:
        return empty
    if isinstance(result, (list, tuple)):
        lines = []
        for i, item in enumerate(result, 1):
            lines.append("%d. %s" % (i, json.dumps(item, default=str, ensure_ascii=False)
                                     if isinstance(item, (dict, list)) else str(item)))
        return cap("\n".join(lines))
    if isinstance(result, dict):
        return cap(json.dumps(result, default=str, ensure_ascii=False, indent=1))
    return cap(str(result))


def workspace_post(content: str, kind: str = "fact", task_id: str | None = None, agent: str | None = None,
                   confidence: float = 0.5, verified: bool = False, sources=None, field: str | None = None) -> str:
    ws, err = _workspace()
    if err:
        return err
    if not str(content or "").strip():
        raise ValueError("content is required")
    task = _task(task_id)
    # Trust boundary: an agent can never mark its own entry verified. Only out-of-band pipelines or
    # `python -m agentkit.approve` set verified. The agent's claim is kept as claimed_verified when the backend has
    # that field, otherwise as the same content tag agentkit uses.
    kw = {}
    if verified:
        if _accepts(ws.post, "claimed_verified"):
            kw["claimed_verified"] = True
        else:
            content = CLAIM_TAG + str(content)
    eid = ws.post(task, agent or os.environ.get("MESH_AGENT_NAME", "agent"), kind, content,
                  confidence=float(confidence), verified=False, sources=_list(sources), field=field, **kw)
    if eid is None:
        return "not stored: the content was empty after scrubbing or contained a secret"
    return "posted entry %s to workspace task %s" % (eid, task)


def workspace_read(task_id: str | None = None, query: str | None = None, k: int = 8) -> str:
    ws, err = _workspace()
    if err:
        return err
    return _render(ws.read(_task(task_id), query=query, k=int(k or 8)),
                   "workspace task %s is empty" % _task(task_id))


def memory_search(text: str, field: str | None = None, k: int = 3) -> str:
    ws, err = _workspace()
    if err:
        return err
    return _render(ws.search(text, field=field, k=int(k or 3)), "no matching memory")


def analogies(text: str, k: int = 3, field: str | None = None) -> str:
    ws, err = _workspace()
    if err:
        return err
    return _render(ws.analogies(text, k=int(k or 3), field=field), "no analogies found")


def workspace_observe(ids, passed: bool) -> str:
    ws, err = _workspace()
    if err:
        return err
    ids = _list(ids) or []
    if not ids:
        raise ValueError("ids is required (workspace entry ids, or card ids like 'card:...')")
    ws.observe([int(i) if isinstance(i, str) and i.isdigit() else i for i in ids], bool(passed))
    return "recorded outcome passed=%s for %d item(s)" % (bool(passed), len(ids))


def workspace_stats() -> str:
    ws, err = _workspace()
    if err:
        return err
    return _render(ws.stats(), "workspace is empty")
