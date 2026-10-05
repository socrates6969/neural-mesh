"""R113: ORT (python) fp32 / dynamic-int8 MiniLM latency on the SAME 192 prompts as src/bin/n19_vnni.rs (corpus A, i*37 % N, first 12 words), medians of 5 interleaved passes."""
import json, os, sys, time, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType
M = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2"); out = sys.argv[1]
q8 = os.path.join(out, "minilm_q8.onnx")
if not os.path.exists(q8): quantize_dynamic(M + "/model.onnx", q8, weight_type=QuantType.QInt8)
texts = json.load(open(os.path.join(os.path.dirname(__file__), "../../results/h10_texts_A.json")))
qs = [" ".join(texts[(i * 37) % len(texts)].split()[:12]) for i in range(192)]
tok = Tokenizer.from_file(M + "/tokenizer.json"); tok.no_padding(); tok.enable_truncation(max_length=128)
enc = tok.encode_batch(qs); print("tokens/query mean %.1f p50 %d max %d" % (np.mean([len(e.ids) for e in enc]), np.median([len(e.ids) for e in enc]), max(len(e.ids) for e in enc)))
def sess(path, th):
    so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1; return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
def run(s, bs):
    names = [i.name for i in s.get_inputs()]; t0 = time.perf_counter(); n = 0
    for st in range(0, 96, bs):
        e = enc[st:st + bs]; L = max(len(x.ids) for x in e); ids = np.zeros((len(e), L), np.int64); am = np.zeros_like(ids)
        for r, x in enumerate(e): ids[r, :len(x.ids)] = x.ids; am[r, :len(x.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in names: feed["token_type_ids"] = np.zeros_like(ids)
        hs = s.run(None, feed)[0]; m = am[:, :, None].astype(np.float32); p = (hs * m).sum(1) / m.sum(1); p /= np.linalg.norm(p, axis=1, keepdims=True); n += len(e)
    return (time.perf_counter() - t0) * 1e3 / n
for th in (1, 4):
    S = {"ORT fp32": sess(M + "/model.onnx", th), "ORT int8": sess(q8, th)}
    for bs in (1, 16):
        res = {k: [] for k in S}
        for _ in range(5):
            for k, s in S.items(): res[k].append(run(s, bs))
        print(f"threads {th} bs={bs}: " + " | ".join(f"{k} {np.median(v):.3f} ms/q" for k, v in res.items()))
