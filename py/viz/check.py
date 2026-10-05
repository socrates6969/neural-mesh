"""verify the rendered HTML against the sources (independent parse of the HTML with regex; compares every cell/edge/node/doc)."""
import json, re, sys, html
import numpy as np
h = open(sys.argv[1], encoding="utf-8").read(); state = json.load(open(sys.argv[2])); brain = json.load(open(sys.argv[3]))
cells = re.findall(r'class="cell" data-f="([^"]*)" data-a="([^"]*)" data-v="([^"]*)"', h)
exp = 0; got = {}
for f, a, v in cells: got.setdefault((html.unescape(f), html.unescape(a)), []).append(float(v))
rows = state["mesh"]["leader"]["weights"]; ok = True
for f, w in rows.items():
    for j, a in enumerate(state["agents"]):
        want = w[j] if j < len(w) else 0.0; exp += 1
        if not any(abs(x - want) < 1e-5 for x in got.get((f, a), [])): ok = False; print("MISMATCH", f, a, want)
for w in brain["weights"]:
    for j, a in enumerate(brain["agents"]):
        exp += 1
        if not any(abs(x - w["weights"][j]) < 1e-5 for x in got.get((w["feature"], a), [])): ok = False; print("MISMATCH brain", w["feature"], a)
edges = re.findall(r'class="edge" data-a="([^"]*)" data-b="([^"]*)" data-w="([^"]*)"', h)
E = state["mesh"]["edges"]; n = len(state["agents"]); want_e = sum(1 for i in range(n) for j in range(i + 1, n) if max(E[i][j], E[j][i]) > 1e-9)
nodes = re.findall(r'class="node" data-a="([^"]*)" data-h="([^"]*)"', h)
docs = re.findall(r'class="doc" data-i="(\d+)"', h); meta = json.loads(re.search(r'id="meta">(.*?)</script>', h).group(1))
print(f"cells found {len(cells)} expected {exp}; edges {len(edges)}/{want_e}; nodes {len(nodes)}/{n}; docs {len(docs)}/{meta.get('docs')}; queries {meta.get('queries')}")
assert ok and len(cells) == exp and len(edges) == want_e and len(nodes) == n and (meta.get("docs") is None or len(docs) == meta["docs"])
assert all(np.isfinite(float(x)) for x in re.findall(r'cx="([-0-9.]+)"', h))
print("VIZ CHECK PASSED")
