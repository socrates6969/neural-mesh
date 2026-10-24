"""Experience memory: a MiniLM neural embedding index of grader-verified solutions, retrieved for new problems.

When an agent gets a problem, the most similar previously-solved problems (cosine similarity of all-MiniLM-L6-v2
sentence embeddings, the same encoder the mesh router uses) are shown with their verified solution, so the agent
can reuse the method. Only pack-free, grader-verified runs are stored; all text is scrubbed.

Live files (the neural mesh): data/lifelong/mesh/experience.jsonl (records) + experience.f32 (384-d vectors).
The successor is train/memory (method cards, hybrid retrieval, daemon, Global Workspace); this module stays as the
MiniLM encoder they share and as a compatible wrapper for existing callers. build_from_results also builds cards.
Built/extended after every training round (Trainer.export_live) and by `python -m train.experience build`.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np

NM = Path(r"C:\Users\mariu\neural-mesh")
LIVE = Path(os.environ.get("NEURAL_MESH_EXPERIENCE", str(NM / "data" / "lifelong" / "mesh" / "experience.jsonl")))
MODEL_DIR = Path(os.environ.get("MESH_MINILM", str(Path.home() / ".ruvector" / "models" / "all-MiniLM-L6-v2")))
MAX_PROBLEM, MAX_SOLUTION = 1500, 2500
_lock = threading.Lock()
_enc = None
_cache: dict = {}


class _Encoder:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(256)
        self.tok.enable_padding(length=None)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], bs: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok.encode_batch([t or " " for t in texts[i:i + bs]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            att = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0]
            m = att[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            out.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


def encoder() -> _Encoder:
    global _enc
    with _lock:
        if _enc is None:
            _enc = _Encoder()
        return _enc


def _vec_path(p: Path) -> Path:
    return p.with_suffix(".f32")


def load(path: Path = LIVE) -> tuple[list[dict], np.ndarray]:
    """Records + vectors, cached per file mtime."""
    if not path.exists() or not _vec_path(path).exists():
        return [], np.zeros((0, 384), np.float32)
    key = (str(path), path.stat().st_mtime_ns, _vec_path(path).stat().st_mtime_ns)
    if _cache.get("key") != key:
        recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        vecs = np.fromfile(_vec_path(path), dtype=np.float32).reshape(-1, 384)
        n = min(len(recs), len(vecs))
        _cache.update(key=key, recs=recs[:n], vecs=vecs[:n])
    return _cache["recs"], _cache["vecs"]


def add(records: list[dict], path: Path = LIVE) -> int:
    """Append new records (dedup by id) and their problem embeddings."""
    old, _ = load(path)
    have = {r["id"] for r in old}
    new = [r for r in records if r["id"] not in have]
    if not new:
        return 0
    vecs = encoder().encode([r["problem"] for r in new])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(_vec_path(path), "ab") as f:
        vecs.astype(np.float32).tofile(f)
    _cache.clear()
    return len(new)


def retrieve(text: str, field: str | None = None, k: int = 2, min_sim: float = 0.55, exclude: set | None = None,
             path: Path = LIVE) -> list[tuple[float, dict]]:
    recs, vecs = load(path)
    if not recs:
        return []
    q = encoder().encode([text[:MAX_PROBLEM]])[0]
    sims = vecs @ q
    out = []
    for i in np.argsort(-sims):
        if sims[i] < min_sim or len(out) >= k:
            break
        r = recs[i]
        if exclude and r.get("task") in exclude:
            continue
        if field and r.get("field") and r["field"] != field:
            continue
        out.append((float(sims[i]), r))
    return out


def render(hits: list[tuple[float, dict]]) -> str:
    if not hits:
        return ""
    parts = ["Experience memory (similar problems solved before; each solution was verified correct). Reuse the method "
             "only where it truly applies:"]
    for j, (s, r) in enumerate(hits, 1):
        parts.append("[%d] (similarity %.2f) Problem:\n%s\n\nVerified solution:\n%s" % (j, s, r["problem"], r["solution"]))
    return "\n\n".join(parts)


def build_from_results(batch: str = "batch1") -> int:
    """Add every pack-free, grader-verified held-out solution of a training batch."""
    from . import data
    from .loop import ROOT
    sys.path.insert(0, str(NM))
    from ingest import scrub as S
    res = [json.loads(l) for l in (ROOT / "train" / batch / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = {}
    for f in {r["field"] for r in res}:
        tasks.update({t["id"]: t for t in data.load(ROOT / "bench", f)})
    best = {}
    for r in res:
        if (r["phase"] == "held_base" and r["passed"] and not r.get("error") and r["task"] in tasks
                and 40 < len(r.get("text") or "") < 4000):
            if r["task"] not in best or len(r["text"]) < len(best[r["task"]]["text"]):
                best[r["task"]] = r
    recs = []
    for tid, r in best.items():
        sp, ss = S.scrub(tasks[tid]["prompt"]), S.scrub(r["text"])
        if sp.secret or ss.secret:
            continue
        recs.append({"id": "exp:" + tid, "task": tid, "field": r["field"], "agent": r["agent"],
                     "problem": sp.text[:MAX_PROBLEM], "solution": ss.text[-MAX_SOLUTION:]})
    n = add(recs)
    try:                      # the shared method-card store (train/memory) grows from the same verified runs
        from .memory.builder import build
        build([batch], log=lambda *a: None)
    except Exception:
        pass
    return n


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        print("experience memory: added", build_from_results(*(sys.argv[2:3] or ["batch1"])), "->", LIVE)