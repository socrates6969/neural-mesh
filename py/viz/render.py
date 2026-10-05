"""neural-mesh visualisation: ONE self-contained HTML (inline SVG, no network, no libraries).
  1. router weights heatmap (feature x agent)   from a CLI state file (--state) or results/brain.json (--brain)
  2. mesh graph: agents as nodes (size = health), edges = Hebbian co-activation weights  (CLI state only)
  3. embedding / retrieval space: PCA(2) of a stored corpus namespace (mesh-memory store or b1 fixtures) with queries and their
     top-1 retrieval (green = correct domain, red = wrong)
Usage: python render.py --state results/viz_demo_state.json --brain results/brain.json --out C:/Users/mariu/experiments/mesh-memory/viz/mesh.html
`selfcheck(html_path, ...)` re-derives the numbers from the sources and checks the HTML carries exactly those values (see py/viz/check.py).
Honest scope: a debugging aid; it renders what is in the files, nothing is inferred."""
import argparse, html, json, math, os, sys
import numpy as np

PAL = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]

def esc(s): return html.escape(str(s))

def heatmap(features, agents, W):
    cw, ch, ml, mt = 54, 22, 150, 90
    w, h = ml + cw * len(agents) + 20, mt + ch * len(features) + 20
    mx = max(1e-9, max(max(r) for r in W)) if W else 1.0
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="router weight heatmap" class="chart">']
    for j, a in enumerate(agents):
        out.append(f'<text x="{ml + cw*j + cw/2}" y="{mt-8}" class="lab" transform="rotate(-35 {ml + cw*j + cw/2} {mt-8})">{esc(a)}</text>')
    for i, f in enumerate(features):
        out.append(f'<text x="{ml-8}" y="{mt + ch*i + ch*0.7}" class="lab" text-anchor="end">{esc(f)}</text>')
        for j, a in enumerate(agents):
            v = W[i][j]; t = v / mx
            out.append(f'<rect class="cell" data-f="{esc(f)}" data-a="{esc(a)}" data-v="{v:.6f}" x="{ml + cw*j}" y="{mt + ch*i}" width="{cw-2}" height="{ch-2}" rx="3" style="fill:color-mix(in srgb, var(--accent) {int(100*t)}%, var(--cell0))"><title>{esc(f)} -> {esc(a)}: {v:.4f}</title></rect>')
            if t > 0.5: out.append(f'<text x="{ml + cw*j + cw/2 - 1}" y="{mt + ch*i + ch*0.68}" class="cv" text-anchor="middle">{v:.2f}</text>')
    out.append("</svg>")
    return "\n".join(out)

def graph(agents, edges, health):
    n = len(agents); R = 150; cx = cy = 190
    pos = [(cx + R * math.cos(2 * math.pi * i / n - math.pi / 2), cy + R * math.sin(2 * math.pi * i / n - math.pi / 2)) for i in range(n)]
    mx = max(1e-9, max((edges[i][j] for i in range(n) for j in range(n) if i < j), default=1e-9))
    out = [f'<svg viewBox="0 0 380 380" role="img" aria-label="mesh graph" class="chart">']
    for i in range(n):
        for j in range(i + 1, n):
            w = max(edges[i][j], edges[j][i])
            if w <= 1e-9: continue
            out.append(f'<line class="edge" data-a="{esc(agents[i])}" data-b="{esc(agents[j])}" data-w="{w:.6f}" x1="{pos[i][0]:.1f}" y1="{pos[i][1]:.1f}" x2="{pos[j][0]:.1f}" y2="{pos[j][1]:.1f}" style="stroke-width:{1+7*w/mx:.2f}"><title>{esc(agents[i])} - {esc(agents[j])}: {w:.3f}</title></line>')
    for i in range(n):
        r = 10 + 14 * max(0.0, min(1.0, health[i]))
        out.append(f'<circle class="node" data-a="{esc(agents[i])}" data-h="{health[i]:.4f}" cx="{pos[i][0]:.1f}" cy="{pos[i][1]:.1f}" r="{r:.1f}" style="fill:{PAL[i % len(PAL)]}"><title>{esc(agents[i])} health {health[i]:.2f}</title></circle>')
        out.append(f'<text x="{pos[i][0]:.1f}" y="{pos[i][1] + r + 13:.1f}" class="lab" text-anchor="middle">{esc(agents[i])}</text>')
    out.append("</svg>")
    return "\n".join(out)

def pca2(X):
    mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = (X - mu) @ Vt[:2].T; var = (S[:2] ** 2) / (S ** 2).sum()
    return P, Vt[:2], mu, var

def space(docs_vecs, labels, queries=None):
    """docs_vecs (n,d); labels list[str]; queries: list of (text, vec, gold_label)"""
    P, V, mu, var = pca2(docs_vecs)
    allp = P if not queries else np.vstack([P] + [((q[1] - mu) @ V.T)[None] for q in queries])
    lo, hi = allp.min(0), allp.max(0); W, H, pad = 640, 460, 24
    sx = lambda x: pad + (x - lo[0]) / (hi[0] - lo[0] + 1e-9) * (W - 2 * pad)
    sy = lambda y: H - pad - (y - lo[1]) / (hi[1] - lo[1] + 1e-9) * (H - 2 * pad)
    labs = sorted(set(labels)); col = {l: PAL[i % len(PAL)] for i, l in enumerate(labs)}
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="embedding space PCA" class="chart">']
    nq = len(queries or [])
    for qi, (qt, qv, gold) in enumerate(queries or []):
        qp = (qv - mu) @ V.T; top = int(np.argmax(docs_vecs @ qv)); ok = labels[top] == gold
        out.append(f'<line class="link {"ok" if ok else "bad"}" x1="{sx(qp[0]):.1f}" y1="{sy(qp[1]):.1f}" x2="{sx(P[top][0]):.1f}" y2="{sy(P[top][1]):.1f}"/>')
    for i, p in enumerate(P):
        out.append(f'<circle class="doc" data-i="{i}" data-l="{esc(labels[i])}" cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="4.2" style="fill:{col[labels[i]]}"><title>{esc(labels[i])} #{i}</title></circle>')
    for qi, (qt, qv, gold) in enumerate(queries or []):
        qp = (qv - mu) @ V.T
        out.append(f'<path class="q" d="M{sx(qp[0]):.1f} {sy(qp[1]) - 5:.1f} l5 10 h-10 z"><title>{esc(qt)} ({esc(gold)})</title></path>')
    out.append("</svg>")
    leg = "".join(f'<span class="lg"><i style="background:{col[l]}"></i>{esc(l)}</span>' for l in labs)
    return "\n".join(out), leg, var, nq

CSS = """:root{--bg:#fff;--fg:#1b1f24;--muted:#5b6670;--accent:#2f6fdb;--cell0:#eef1f5;--card:#f7f8fa;--ok:#2a9d4b;--bad:#d1453b;--edge:#8a96a3}
@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#13161a;--fg:#e6e9ee;--muted:#9aa5b1;--accent:#6aa3ff;--cell0:#1d222a;--card:#1a1e24;--edge:#6c7886}}
:root[data-theme=dark]{--bg:#13161a;--fg:#e6e9ee;--muted:#9aa5b1;--accent:#6aa3ff;--cell0:#1d222a;--card:#1a1e24;--edge:#6c7886}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:980px;margin:0 auto;padding:20px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 6px}p.s{color:var(--muted);margin:0 0 10px}
.card{background:var(--card);border-radius:10px;padding:12px;overflow-x:auto}.chart{width:100%;height:auto;max-width:860px}
.lab{font-size:11px;fill:var(--fg)}.cv{font-size:10px;fill:#fff;pointer-events:none}.edge{stroke:var(--edge);stroke-linecap:round;opacity:.8}
.node{stroke:var(--bg);stroke-width:2}.doc{stroke:var(--bg);stroke-width:1}.q{fill:var(--fg)}.link.ok{stroke:var(--ok);stroke-width:1.4;opacity:.7}.link.bad{stroke:var(--bad);stroke-width:1.8}
.lg{display:inline-flex;align-items:center;margin:2px 10px 2px 0;font-size:12px}.lg i{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px}"""

def build(state=None, brain=None, space_args=None, title="neural-mesh"):
    parts = []; meta = {}
    if state:
        s = json.load(open(state)); agents = s["agents"]; rows = s["mesh"]["leader"]["weights"]; feats = sorted(rows)
        W = [[rows[f][j] if j < len(rows[f]) else 0.0 for j in range(len(agents))] for f in feats]
        parts.append(("Router weights (state file)", f"{len(feats)} features x {len(agents)} agents; brightness = weight; the Hebbian row for a feature is its learned preference over agents.", heatmap(feats, agents, W)))
        parts.append(("Mesh graph", "node size = health EMA; edge width = learned co-activation (teams that succeeded together).", graph(agents, s["mesh"]["edges"], s["mesh"]["health"])))
        meta.update(heat_cells=len(feats) * len(agents), nodes=len(agents))
    if brain:
        b = json.load(open(brain)); feats = [w["feature"] for w in b["weights"]]
        parts.append((f"Router weights (brain export: {b['router']})", esc(b["world"]), heatmap(feats, b["agents"], [w["weights"] for w in b["weights"]])))
        meta.update(brain_cells=len(feats) * len(b["agents"]))
    if space_args:
        X, labels, queries = space_args; svg, leg, var, nq = space(X, labels, queries)
        parts.append(("Embedding space (PCA, 2 components: %.0f%% + %.0f%% of variance)" % (100 * var[0], 100 * var[1]), "dots = documents coloured by domain; triangles = queries; line to the top-1 retrieved document (green = right domain, red = wrong).", svg + f'<div>{leg}</div>'))
        meta.update(docs=len(labels), queries=nq)
    body = "".join(f'<h2>{esc(t) if not t.startswith("Router weights (brain") else esc(t)}</h2><p class="s">{d}</p><div class="card">{svg}</div>' for t, d, svg in parts)
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><p class="s">Rendered from files; nothing inferred. Synthetic data only.</p>{body}</main><script type="application/json" id="meta">{json.dumps(meta)}</script></body></html>', meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--state"); ap.add_argument("--brain"); ap.add_argument("--space", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); sa = None
    if a.space:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b1"))
        from lib import Data
        D = Data(); dd = D.stores["domain"]; X = np.stack([D.v(d) for d in dd["docs"]]); labels = dd["dom"]
        qs = [(q2, D.v(q2), dom) for _, dom, q1, q2 in dd["qs"][::6]]
        sa = (X, labels, qs)
    h, meta = build(a.state, a.brain, sa)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); open(a.out, "w", encoding="utf-8").write(h); print("wrote", a.out, meta, len(h), "bytes")
