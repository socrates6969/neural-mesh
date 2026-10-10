"""Shared 384-d MiniLM vector space (queen directive "maximise vectors where useful").

Producer: neural-mesh's pure-Rust encoder (src/encoder.rs, all-MiniLM-L6-v2, L2-normalised) through the
``embed_texts`` binary. Every text is embedded ONCE and cached by content hash in ``data/lifelong/vectors/``
(keys.jsonl: {"k": sha16(text), "r": row}; vecs.f32: row-major little-endian f32, 384 per row). Records carry the
key (``vec_key``) instead of the vector. If the encoder or its weights are missing, ``available()`` is False and the
callers fall back (memory -> HashEmbedding; sft near-dup / clusters -> skipped) and say so in the run summary.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .util import append_jsonl, iter_jsonl, sha16

DIM = 384


def vec_key(text: str) -> str:
    return sha16("minilm-l6", text)


class VectorStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir: Path = cfg.data / "vectors"
        self.keys_path = self.dir / "keys.jsonl"
        self.vecs_path = self.dir / "vecs.f32"
        models = cfg.data / "models"
        self.weights = Path(os.environ.get("INGEST_ENCODER_WEIGHTS", str(models / "minilm.f32")))
        self.tokenizer = Path(os.environ.get("INGEST_ENCODER_TOKENIZER", str(models / "tokenizer.json")))
        self.exe = Path(os.environ.get("INGEST_EMBED_EXE", str(cfg.mesh_exe.parent / "embed_texts.exe")))
        self.index: dict[str, int] = {}
        for r in iter_jsonl(self.keys_path):
            if "k" in r and "r" in r:
                self.index[r["k"]] = r["r"]
        rows = (self.vecs_path.stat().st_size // (4 * DIM)) if self.vecs_path.exists() else 0
        self.index = {k: r for k, r in self.index.items() if r < rows}  # a torn tail is simply re-embedded
        self.mat = np.fromfile(self.vecs_path, dtype="<f4", count=rows * DIM).reshape(rows, DIM) if rows else np.zeros((0, DIM), "<f4")
        self.stats = {"embedded": 0, "cached": 0, "error": None}

    def available(self) -> bool:
        return self.exe.exists() and self.weights.exists() and self.tokenizer.exists()

    def ensure(self, texts: list[str]) -> bool:
        """Embed every text not yet cached. Returns False if the encoder is unavailable or failed."""
        missing, seen = [], set()
        for t in texts:
            k = vec_key(t)
            if k in self.index or k in seen:
                continue
            seen.add(k)
            missing.append(t)
        self.stats["cached"] += len(texts) - len(missing)
        if not missing:
            return True
        if not self.available():
            self.stats["error"] = "encoder unavailable"
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp_in, tmp_out = self.dir / f"in{os.getpid()}.json", self.dir / f"out{os.getpid()}.f32"
        try:
            for i in range(0, len(missing), 2000):
                chunk = missing[i : i + 2000]
                tmp_in.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
                r = subprocess.run([str(self.exe), str(self.weights), str(self.tokenizer), str(tmp_in), str(tmp_out)],
                                   capture_output=True, text=True, timeout=3600)
                if r.returncode != 0:
                    self.stats["error"] = f"embed_texts exit {r.returncode}"
                    return False
                v = np.fromfile(tmp_out, dtype="<f4")
                if v.size != len(chunk) * DIM:
                    self.stats["error"] = "embed_texts size mismatch"
                    return False
                base = self.mat.shape[0]
                with open(self.vecs_path, "ab") as f:  # vectors first, keys second: a crash leaves an orphan row, never a dangling key
                    f.write(v.astype("<f4").tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                append_jsonl(self.keys_path, [{"k": vec_key(t), "r": base + j} for j, t in enumerate(chunk)])
                for j, t in enumerate(chunk):
                    self.index[vec_key(t)] = base + j
                self.mat = np.vstack([self.mat, v.reshape(len(chunk), DIM)])
                self.stats["embedded"] += len(chunk)
        finally:
            for p in (tmp_in, tmp_out):
                try:
                    p.unlink()
                except OSError:
                    pass
        return True

    def get(self, text: str) -> np.ndarray | None:
        r = self.index.get(vec_key(text))
        return None if r is None else self.mat[r]

    def matrix(self, texts: list[str]) -> np.ndarray | None:
        rows = [self.index.get(vec_key(t)) for t in texts]
        if any(r is None for r in rows):
            return None
        return self.mat[rows] if rows else np.zeros((0, DIM), "<f4")


def leader_clusters(vecs: np.ndarray, threshold: float) -> list[int]:
    """Greedy leader clustering in the given order: a row joins the first leader with cosine >= threshold, else leads a new cluster."""
    leaders: list[int] = []
    out = []
    for i in range(vecs.shape[0]):
        if leaders:
            sims = vecs[leaders] @ vecs[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                out.append(out[leaders[j]])
                continue
        leaders.append(i)
        out.append(len(leaders) - 1)
    return out
