import sys; sys.path.insert(0, "../")
from lab import *
import lab, pickle, f8_embed
from lib import cosm
from gate import compare
from r22 import DataBGE, D2
import numpy as np
def acc_cloze_mix(D_, items, lam):
    r = []
    for it in items:
        if it["skill"] != "sentence_completion": continue
        st = it["stem"]; O = np.stack([D_.v(o) for o in it["opts"]]); F = np.stack([D_.v(st.replace("___", o)) for o in it["opts"]])
        r.append(int(np.argmax(cosm(D_.v(st), F) + lam * cosm(D_.v(st), O))) == it["ans"])
    return float(np.mean(r))
c = {}
for w in WORLDS:
    for s in ALLS:
        ax = {f"sem_{k}": v for k, v in semantic_acc(D2, D2.items[f"{w}{s}"]).items()}
        ax["sem_sentence_completion_mix"] = acc_cloze_mix(D2, D2.items[f"{w}{s}"], 0.5)
        c[(w, s)] = ax
D2.save()
base = {k: dict(BASE[k]) for k in c}
for k in base: base[k]["sem_sentence_completion_mix"] = base[k]["sem_sentence_completion"]
tgt = [f"sem_{k}" for k in ("vocab","similarities","classification","verbal_analogies","sentence_completion")] + ["sem_sentence_completion_mix"]
v, rows, gain, loss = compare(base, c, tgt, tol=0.002, quiet=True, axes=tgt)
print("BGE on ORIGINAL pools vs MiniLM:", v, "| gains", [g for g in gain] , "| losses", loss)
for ax in tgt: print("  ", ax, " ".join(f"{r['w']}{r['sp'][0]} {r['mean']:+.3f}(p{r['ph']:.2g})" for r in rows if r["ax"] == ax))
