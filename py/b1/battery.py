"""Run the battery for a SYSTEM CONFIG over seeds; returns table[(world,seed)] -> {axis: value}."""
import json, sys, time, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from lib import *

_RW = []
def run_battery(D, seeds, arm="hybrid", alpha=0.7, make=None, E=None, routers=None, with_exec=True, real=False):
    tab = {}
    if real:
        if not _RW: _RW.append(RealWorld("A"))
        for s in seeds:
            a = _RW[0].axes(s, arm, alpha, make)
            if a: tab[("R", s)] = a
    for w in WORLDS:
        for s in seeds:
            ax = {}
            ax.update({f"sem_{k}": v for k, v in semantic_acc(D, D.items[f"{w}{s}"], E).items()})
            ax.update(knowledge_axes(D, w, s, arm, alpha, make))
            if with_exec: ax.update(exec_axes(D, w, s))
            if routers: ax.update(routing_axes(D, w, s, routers))
            tab[(w, s)] = ax
    return tab

def summarize(tab, seeds, axes=None):
    """mean over worlds of per-seed means, per axis"""
    out = {}
    keys = axes or sorted({k for v in tab.values() for k in v})
    for k in keys:
        out[k] = float(np.mean([tab[(w, s)][k] for w in WORLDS + "R" for s in seeds if (w, s) in tab and k in tab[(w, s)]]))
    return out

if __name__ == "__main__":
    D = Data(); t0 = time.time()
    knn = KNNRouter(D, TUNE)
    routers = {"kw": route_kw, "knn": knn}
    out = {}
    for arm in ("dense", "bm25", "hybrid"):
        tab = run_battery(D, TUNE + HELD + CONF, arm=arm, routers=routers if arm == "hybrid" else None, with_exec=(arm == "hybrid"), real=True)
        out[arm] = {f"{w}{s}": v for (w, s), v in tab.items()}
        print(arm, "done", round(time.time() - t0))
    json.dump(out, open(f"{RES}/b1_baseline.json", "w"))
    for name, sd in (("TUNE", TUNE), ("HELD", HELD), ("CONF", CONF)):
        print("==", name)
        for arm in ("dense", "bm25", "hybrid"):
            tab = {(k[0], int(k[1:])): v for k, v in out[arm].items()}
            sm = summarize(tab, sd)
            print(arm, {k: round(v, 3) for k, v in sm.items() if not k.startswith(("exec_", "claim_", "us_"))})
