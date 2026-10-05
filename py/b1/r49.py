"""R49: BGE-base (opt-in quality profile) on the V2 verbal battery (vocab / analogies / cloze, and cloze_mix) vs MiniLM default."""
import sys; sys.path.insert(0, "../")
from v2 import *
import v2, pickle, f8_embed
BGE = f8_embed.Embedder(f8_embed.BGE, "cls", 768)
BC = os.path.join(RES, "v2_bge.pkl"); _B = pickle.load(open(BC, "rb")) if os.path.exists(BC) else {}
def bemb(t):
    if t not in _B: bembed_many([t])
    return _B[t]
def bembed_many(ts):
    ts = [t for t in dict.fromkeys(ts) if t not in _B]
    for i in range(0, len(ts), 64):
        vs = BGE.batch(ts[i:i + 64])
        for t, x in zip(ts[i:i + 64], vs): _B[t] = x
    pickle.dump(_B, open(BC, "wb"))
def pred_bge(sk, x): return base_pred(sk, x, bemb)
def pred_bge_mix(sk, x):
    if sk != "cloze": return base_pred(sk, x, bemb)
    F = np.stack([bemb(x["st"].replace("___", o)) for o in x["opts"]]); O = np.stack([bemb(o) for o in x["opts"]]); s = bemb(x["st"])
    return int(np.argmax(cosm(s, F) + 0.5 * cosm(s, O)))
def pred_mini_mix(sk, x):
    if sk != "cloze": return base_pred(sk, x)
    F = np.stack([emb(x["st"].replace("___", o)) for o in x["opts"]]); O = np.stack([emb(o) for o in x["opts"]]); s = emb(x["st"])
    return int(np.argmax(cosm(s, F) + 0.5 * cosm(s, O)))
if __name__ == "__main__":
    t0 = time.time(); seeds = TUNE + HELD + CONF
    B = pickle.load(open(os.path.join(RES, "v2_base.pkl"), "rb"))
    ts = all_texts(base_texts, seeds); bembed_many(ts); print("bge embedded", len(_B), round(time.time() - t0))
    for name, p in (("bge", pred_bge), ("bge+mix", pred_bge_mix)):
        c = run_v2(p, base_texts)
        gate(B, c, ["v2_vocab", "v2_analogies", "v2_cloze"], name=f"R49 {name} vs MiniLM default (V2 verbal)")
    cm = run_v2(pred_mini_mix, base_texts)
    c2 = run_v2(pred_bge_mix, base_texts)
    gate(cm, c2, ["v2_cloze"], name="R49c bge+mix vs MiniLM+mix (cloze)", )
