"""384-d all-MiniLM-L6-v2 sentence vectors (the same ONNX + tokenizer the mesh's Rust encoder (src/encoder.rs, R101/R109
parity) is exported from: ~/.ruvector/models/all-MiniLM-L6-v2), run with onnxruntime on CPU. Used for: front-end
retrieval, OOD abstain (max cosine to the trained set), and nearest-but-wrong judge decoys."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODEL_DIR = Path(os.environ.get("MESH_MINILM", Path.home() / ".ruvector" / "models" / "all-MiniLM-L6-v2"))
MAXLEN = 128


class Encoder:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(MAXLEN)
        self.tok.enable_padding(length=None)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
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
            h = self.sess.run(None, feed)[0]                       # [b, L, 384]
            m = att[..., None].astype(np.float32)
            v = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)       # mean pool
            out.append(v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12))
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 384), np.float32)


def max_cos(q: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """per query: max cosine to any row of bank (both L2-normalised); 0 when the bank is empty."""
    if len(bank) == 0:
        return np.zeros(len(q), np.float32)
    return (q @ bank.T).max(1)


def nearest_wrong(i: int, vecs: np.ndarray, answers: list[str]) -> int:
    """index of the most similar OTHER item whose reference answer differs (hard decoy for judge calibration); -1 if none."""
    s = vecs @ vecs[i]
    for j in np.argsort(-s):
        if j != i and answers[j].strip() != answers[i].strip():
            return int(j)
    return -1
