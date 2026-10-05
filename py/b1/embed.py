import json, os, sys, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
M = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")
_C = {}
def embed(texts, bs=64):
    if not _C:
        tok = Tokenizer.from_file(os.path.join(M, "tokenizer.json")); tok.no_padding(); tok.enable_truncation(max_length=128)
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        _C["tok"] = tok; _C["sess"] = ort.InferenceSession(os.path.join(M, "model.onnx"), so, providers=["CPUExecutionProvider"])
    tok, sess = _C["tok"], _C["sess"]
    names = [i.name for i in sess.get_inputs()]
    enc = tok.encode_batch(texts); order = sorted(range(len(texts)), key=lambda i: len(enc[i].ids))
    out = np.zeros((len(texts), 384), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s+bs]; L = max(len(enc[i].ids) for i in idx)
        ids = np.zeros((len(idx), L), np.int64); am = np.zeros((len(idx), L), np.int64)
        for r, i in enumerate(idx):
            n = len(enc[i].ids); ids[r, :n] = enc[i].ids; am[r, :n] = 1
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in names: feed["token_type_ids"] = np.zeros_like(ids)
        hs = sess.run(None, feed)[0]; m = am[:, :, None].astype(np.float32)
        p = (hs * m).sum(1) / np.clip(m.sum(1), 1e-9, None); p /= np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
        for r, i in enumerate(idx): out[i] = p[r]
    return out
if __name__ == "__main__":
    t = json.load(open(os.path.join(D, "b1_texts.json")))
    e = embed(t); e.tofile(os.path.join(D, "b1_emb.f32")); print(e.shape)
