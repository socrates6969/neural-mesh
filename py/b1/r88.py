"""R88: would the mesh's tool routes (calc/units/csp/sandbox) have caught any REAL delegated-agent traffic? Source: first user message (= the delegated task prompt) of every recorded
subagent transcript + human prompts of main sessions on this machine. Local, in-process only; AGGREGATE rates are printed/saved, no prompt text is stored. kNN router k=5 (battery seeds 1-30 +
72 ROUTE3 exemplars, as R79/R80), first 400 chars of each prompt. Ground truth is NOT labelled: tasks here are coding/research delegations, so any tool-route hit is counted as a FALSE tool route (an upper bound:
a prompt that contained a genuine arithmetic request would be mislabelled as false)."""
import sys, os, json, glob, collections; sys.path.insert(0, "../")
from r79 import *
from r80 import knn_conf, X1, y1
import numpy as np
def first_user_text(f):
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") == "user" and not o.get("isMeta"):
            c = o.get("message", {}).get("content")
            if isinstance(c, str): return c
            if isinstance(c, list):
                t = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                if t.strip(): return t
    return None
def human_prompts(f, cap=40):
    out = []
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") == "user" and not o.get("isSidechain") and not o.get("isMeta"):
            c = o.get("message", {}).get("content")
            if isinstance(c, str) and len(c) > 3 and not c.lstrip().startswith("<"): out.append(c)
    return out[:cap]
fs = glob.glob(os.path.expanduser("~/.claude/projects/*/**/*.jsonl"), recursive=True)
sub = [first_user_text(f) for f in fs if "/subagents/" in f.replace("\\", "/")]; sub = [t for t in sub if t]
hum = [t for f in fs if "/subagents/" not in f.replace("\\", "/") for t in human_prompts(f)]
F = knn_conf(X1, y1); TOOLS_ = {"calc", "units", "csp", "sandbox"}
for name, P in (("delegated subagent prompts", sub), ("human prompts (main sessions)", hum)):
    res = [F(t[:400]) for t in P]; n = len(P)
    lens = [1.33 * len(t.split()) for t in P]
    labs = collections.Counter(l for l, _, _ in res)
    print(f"{name}: n={n}, median tokens(proxy, full text) {np.median(lens):.0f}; kNN label shares: " + ", ".join(f"{k} {v/n:.3f}" for k, v in labs.most_common()))
    for tau in (0, .6, .8):
        tool = sum(1 for l, sh, sm in res if l in TOOLS_ and sh >= tau); print(f"   tau={tau}: routed to a LOCAL TOOL (false, by assumption) {tool}/{n} = {tool/n:.4f}")
    for smin in (.4, .5, .6):
        tool = sum(1 for l, sh, sm in res if l in TOOLS_ and sh >= .6 and sm >= smin); print(f"   tau=.6 + best-sim floor {smin}: routed to a LOCAL TOOL {tool}/{n}")
    print(f"   best-similarity to any router exemplar: median {np.median([sm for _,_,sm in res]):.3f}, p90 {np.percentile([sm for _,_,sm in res],90):.3f}  (in-distribution battery items are ~.7-1.0)")
D.save_extra()
