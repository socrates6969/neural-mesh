"""Test-only workspace backends for the hook subprocess tests (selected with NEURAL_MESH_WORKSPACE_MODULE)."""
import json
import os
import time


def read(task_id, query=None, k=8):
    time.sleep(float(os.environ.get("SLOW_WS_SLEEP", "10")))
    return [{"id": "late", "content": "too late", "task_id": task_id}]


def search(text, field=None, k=3):
    return read(None)


def analogies(text, k=3):
    return read(None)


def post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None):
    path = os.environ.get("SLOW_WS_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_id": task_id, "agent": agent, "kind": kind, "content": content,
                                 "verified": verified}) + "\n")
    time.sleep(float(os.environ.get("SLOW_WS_POST_SLEEP", "0")))
    return "p1"