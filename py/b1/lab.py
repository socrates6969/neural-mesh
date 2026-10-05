"""round runner: python -c / import; run(name, target, **cand_kwargs) -> compares to baseline hybrid table on all seeds, logs jsonl."""
from rounds import *
from gate import compare
import json, time
BASE = base_tab()
def run(name, target, tune=None, **kw):
    t0 = time.time(); real = kw.pop("real", True)
    c = cand(real=real, **kw)
    print(f"=== {name}  ({time.time()-t0:.0f}s)")
    v, rows, gain, loss = compare(BASE, c, target, quiet=True)
    print("  verdict:", v, "| target gains:", [g for g in gain if g[0] in target] or "none", "| losses:", loss or "none")
    # compact target table
    for ax in target:
        for w in WORLDS + "R":
            rr = [r for r in rows if r["ax"] == ax and r["w"] == w]
            if rr: print(f"   {ax:14s}{w} " + " | ".join(f"{r['sp']} d={r['mean']:+.4f} ph={r['ph']:.2g}" for r in rr))
    with open(f"{RES}/b1_rounds.jsonl", "a") as f: f.write(json.dumps(dict(name=name, verdict=v, gains=gain, losses=loss, target=target)) + "\n")
    return v, c
