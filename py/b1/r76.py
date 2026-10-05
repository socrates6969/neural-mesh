"""R76: tokens SENT TO THE LLM per seed-world, as a first-class gate axis, for routing policies (all-LLM / keyword rules / MiniLM kNN).
Items = routing_items (battery question text + true route; tool routes = calc/units/csp/sandbox are answered by the local verified tools at ~0 LLM tokens, exec_* axes are 1.0).
Axes: tok_llm = question tokens (1.33 x words) forwarded to the LLM per item (completion tokens ignored: only the avoidable prompt is counted -> a LOWER BOUND on savings);
tool_ok = 1 - fraction of NON-tool items wrongly answered by a tool (quality); tool_catch = fraction of tool items kept away from the LLM (realised saving)."""
from lab import *
from lib import *
from gate import compare
TOOLS = {"calc", "units", "csp", "sandbox"}
D = Data(); knn = KNNRouter(D, TUNE)
def axes(world, seed, router):
    items = routing_items(D, world, seed); tk = [1.33 * len(t.split()) for t, _ in items]
    if router is None: return {"tok_llm": float(np.mean(tk)), "tool_ok": 1.0, "tool_catch": 0.0, "route": 0.0}
    pred = [router(t) for t, _ in items]; send = [p not in TOOLS for p in pred]
    nt = [lab not in TOOLS for _, lab in items]
    tw = [(p in TOOLS) for p, n in zip(pred, nt) if n]; tm = [(p not in TOOLS) for p, n in zip(pred, nt) if not n]
    return {"tok_llm": float(np.mean([t if s else 0 for t, s in zip(tk, send)])), "tool_ok": 1 - float(np.mean(tw)), "tool_catch": 1 - float(np.mean(tm)), "route": float(np.mean([p == lab for p, (_, lab) in zip(pred, items)]))}
def table(router, seeds): return {(w, s): axes(w, s, router) for w in WORLDS for s in seeds}
if __name__ == "__main__":
    seeds = HELD + CONF; T = {"llm": table(None, seeds), "kw": table(route_kw, seeds), "knn": table(knn, seeds)}
    for n, t in T.items():
        print(f"{n:4s} " + " ".join(f"{k} {np.mean([v[k] for v in t.values()]):.3f}" for k in ("tok_llm", "tool_ok", "tool_catch", "route")))
    for cand, base in (("kw", "llm"), ("knn", "llm"), ("knn", "kw")):
        print(f"== {cand} vs {base}"); compare(T[base], T[cand], ["tok_llm"], tol=0.002, delta=0.01, quiet=False)
