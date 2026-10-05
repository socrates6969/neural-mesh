"""R101: export all-MiniLM-L6-v2 weights from the ONNX file to one flat f32 file for the pure-Rust encoder (src/encoder.rs).
Layout: word_emb[30522x384], pos_emb[512x384], type0[384], embLN w,b; per layer (6): Wq,bq,Wk,bk,Wv,bv,Wo,bo,LN1w,LN1b,W1,b1,W2,b2,LN2w,LN2b; matrices stored [out][in]."""
import onnx, os, sys, numpy as np
from onnx import numpy_helper
M = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2"); out = sys.argv[1]
m = onnx.load(M + "/model.onnx"); I = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
nodes = {n.name: n for n in m.graph.node}
def mm(name):
    n = nodes[name]; w = [x for x in n.input if x in I][0]; return I[w].T.copy()  # [out,in]
parts = [I["embeddings.word_embeddings.weight"], I["embeddings.position_embeddings.weight"], I["embeddings.token_type_embeddings.weight"][0], I["embeddings.LayerNorm.weight"], I["embeddings.LayerNorm.bias"]]
for l in range(6):
    p = f"encoder.layer.{l}."; b = f"/encoder/layer.{l}/"
    for nm, key in (("attention/self/query", "attention.self.query"), ("attention/self/key", "attention.self.key"), ("attention/self/value", "attention.self.value"), ("attention/output/dense", "attention.output.dense")):
        parts += [mm(b + nm + "/MatMul"), I[p + key + ".bias"]]
    parts += [I[p + "attention.output.LayerNorm.weight"], I[p + "attention.output.LayerNorm.bias"]]
    parts += [mm(b + "intermediate/dense/MatMul"), I[p + "intermediate.dense.bias"], mm(b + "output/dense/MatMul"), I[p + "output.dense.bias"], I[p + "output.LayerNorm.weight"], I[p + "output.LayerNorm.bias"]]
flat = np.concatenate([np.asarray(x, np.float32).ravel() for x in parts]); flat.tofile(out); print(flat.size, "floats", flat.nbytes / 1e6, "MB")
