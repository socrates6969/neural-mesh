"""Adapter for the shared Global Workspace (train.memory.workspace, built by mesh-memory on exp/shared-memory).

Every call degrades to "nothing" (None / []) when the backend is missing, disabled (NEURAL_MESH_WORKSPACE=off) or
raises. Results are normalized to plain dicts: {id, kind, content, agent, confidence, verified, task_id, score, field}.
Tests swap the backend with set_backend(obj); NEURAL_MESH_WORKSPACE_MODULE=pkg.mod loads another module instead
(e.g. a daemon client).
"""
from __future__ import annotations

import os

_UNSET = object()
_backend = _UNSET


def set_backend(obj) -> None:
    """Use obj (anything with post/read/search/analogies/observe) as the backend; None disables; reset() undoes."""
    global _backend
    _backend = obj


def reset() -> None:
    global _backend
    _backend = _UNSET


def backend():
    global _backend
    if os.environ.get("NEURAL_MESH_WORKSPACE", "").strip().lower() == "off":
        return None
    if _backend is _UNSET:
        try:
            import importlib
            _backend = importlib.import_module(os.environ.get("NEURAL_MESH_WORKSPACE_MODULE") or "train.memory.workspace")
        except Exception:
            _backend = None
    return _backend


def _call(name, *a, **kw):
    b = backend()
    fn = getattr(b, name, None) if b is not None else None
    if not callable(fn):
        return None
    try:
        return fn(*a, **kw)
    except Exception:
        return None


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def item(x) -> dict | None:
    if isinstance(x, str):
        x = {"content": x}
    elif isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[1], dict):
        x = dict(x[1], score=x[0])                       # (score, record) pairs
    if not isinstance(x, dict):
        return None
    content = x.get("content") or x.get("text") or x.get("body") or x.get("card") or ""
    if not str(content).strip():
        return None
    return {"id": str(x.get("id") or x.get("card_id") or "")[:64], "kind": str(x.get("kind") or "")[:20],
            "content": str(content), "agent": str(x.get("agent") or "")[:80],
            "confidence": _num(x.get("confidence")), "verified": x.get("verified") is True,
            "task_id": str(x.get("task_id") or ""), "score": _num(x.get("score")),
            "field": str(x.get("field") or "")}


def _items(v) -> list[dict]:
    if isinstance(v, dict):
        v = v.get("items") or v.get("hits") or v.get("results") or []
    if not isinstance(v, (list, tuple)):
        return []
    return [i for i in (item(x) for x in v) if i]


def post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None):
    return _call("post", task_id, agent, kind, content, confidence=confidence, verified=verified, sources=sources,
                 field=field)


def read(task_id, query=None, k=8) -> list[dict]:
    return _items(_call("read", task_id, query=query, k=k))


def search(text, field=None, k=3) -> list[dict]:
    return _items(_call("search", text, field=field, k=k))


def analogies(text, k=3) -> list[dict]:
    return _items(_call("analogies", text, k=k))


def observe(ids, passed) -> None:
    _call("observe", list(ids), bool(passed))