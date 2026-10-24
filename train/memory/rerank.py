"""Small ONNX cross-encoder (cross-encoder/ms-marco-MiniLM-L6-v2) that scores (query, card) pairs jointly."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import config as C

_lock = threading.Lock()
_ce = None


class CrossEncoder:
    def __init__(self, model_dir: Path = C.CROSS, max_len: int = 320):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.enable_truncation(max_len, strategy="longest_first")
        self.tok.enable_padding(length=None)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, np.float32)
        enc = self.tok.encode_batch([(query or " ", p or " ") for p in passages])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        return self.sess.run(None, feed)[0].reshape(-1).astype(np.float32)


def available(model_dir: Path = C.CROSS) -> bool:
    return (model_dir / "model.onnx").exists() and (model_dir / "tokenizer.json").exists()


def get() -> CrossEncoder | None:
    global _ce
    with _lock:
        if _ce is None and available():
            _ce = CrossEncoder()
        return _ce