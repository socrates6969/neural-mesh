r"""Global Workspace tools: thin wrappers over train.memory.workspace (built on exp/shared-memory by mesh-memory).

The module is imported from MESH_WORKSPACE_ROOT (default C:\Users\mariu\neural-mesh). Until it exists every tool
returns a clear "workspace not available" message instead of failing. Only a successful import is cached.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

from tools.mesh_mcp.common import cap

DEFAULT_ROOT = r"C:\Users\mariu\neural-mesh"
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


def _task(task_id):
    return task_id or os.environ.get("CLAUDE_SESSION_ID") or "default"


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


def workspace_post(content: str, kind: str = "note", task_id: str | None = None, agent: str | None = None,
                   confidence: float = 0.5, verified: bool = False, sources=None, field: str | None = None) -> str:
    ws, err = _workspace()
    if err:
        return err
    if not str(content or "").strip():
        raise ValueError("content is required")
    r = ws.post(_task(task_id), agent or os.environ.get("MESH_AGENT_NAME", "agent"), kind, content,
                confidence=float(confidence), verified=bool(verified), sources=sources, field=field)
    return _render(r, "posted to workspace task %s" % _task(task_id))


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


def analogies(text: str, k: int = 3) -> str:
    ws, err = _workspace()
    if err:
        return err
    return _render(ws.analogies(text, k=int(k or 3)), "no analogies found")
