"""Static mesh-ops dashboard: one self-contained HTML file plus status.json.

    py -3.12 ops/dashboard.py [--out DIR] [--nm-root DIR] [--ac-root DIR]

Reads every source read-only (see ops/sources.py), writes <out>/index.html and <out>/status.json atomically.
No external resources: inline CSS and inline SVG only, so the file opens offline from disk.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import sources  # noqa: E402

CSS = """
:root{--bg:#0f1216;--panel:#171b21;--line:#2a313b;--fg:#d8dee6;--dim:#8b95a3;--ok:#3fb950;--warn:#d29922;
--err:#f85149;--base:#6e7681;--pack:#58a6ff;--accent:#bc8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 "Segoe UI",system-ui,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}h2{font-size:15px;margin:0 0 10px}h3{font-size:13px;margin:12px 0 6px;color:var(--dim)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:14px;padding:14px 20px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px;min-width:0;overflow-x:auto}
section.wide{grid-column:1/-1}table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.dim{color:var(--dim)}.ok{color:var(--ok)}.warn{color:var(--warn)}.err{color:var(--err)}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;border:1px solid currentColor}
.kpis{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:8px}.kpi b{display:block;font-size:20px}
.kpi span{color:var(--dim);font-size:12px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;
background:#0b0e12;border:1px solid var(--line);border-radius:6px;padding:8px;margin:6px 0 0;max-height:260px;overflow:auto}
.legend span{margin-right:12px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}
svg text{fill:var(--dim);font-size:11px}
"""


def e(x) -> str:
    return html.escape("" if x is None else str(x))


def pill(level: str, text: str | None = None) -> str:
    return '<span class="pill %s">%s</span>' % (level, e(text or level))


def status_level(s: str | None) -> str:
    return {"promoted": "ok", "candidate": "warn", "none": "dim", None: "dim"}.get(s, "dim")


def rate(pair) -> float | None:
    try:
        return pair[0] / pair[1] if pair and pair[1] else None
    except (TypeError, IndexError):
        return None


def fmt_rate(pair) -> str:
    r = rate(pair)
    return "%d/%d (%.0f%%)" % (pair[0], pair[1], 100 * r) if r is not None else "-"


def fmt_p(p) -> str:
    return "-" if p is None else ("%.3f" % p if p >= 0.001 else "%.1e" % p)


def age(h: float | None) -> str:
    if h is None:
        return "-"
    return "%.0f min" % (h * 60) if h < 1 else ("%.1f h" % h if h < 48 else "%.1f d" % (h / 24))


def size(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return ("%d %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024
    return "-"


# ---------------------------------------------------------------- charts (inline SVG)

def bars_pair(rows: list[tuple[str, float | None, float | None]], width: int = 520) -> str:
    """Grouped horizontal bars: base (grey) vs pack (blue), 0..100 %."""
    if not rows:
        return '<p class="dim">no data</p>'
    lab, bar_h, gap = 210, 7, 6
    h = len(rows) * (2 * bar_h + gap) + 18
    scale = width - lab - 40
    out = ['<svg viewBox="0 0 %d %d" width="%d" style="max-width:100%%" role="img" aria-label="held-out base vs pack">' % (width, h, width)]
    for i, (name, b, p) in enumerate(rows):
        y = i * (2 * bar_h + gap)
        out.append('<text x="0" y="%d">%s</text>' % (y + bar_h + 4, e(name[:34])))
        for j, (v, col) in enumerate(((b, "var(--base)"), (p, "var(--pack)"))):
            if v is None:
                continue
            out.append('<rect x="%d" y="%d" width="%.1f" height="%d" fill="%s" rx="1"/>'
                       % (lab, y + j * bar_h, max(1, v * scale), bar_h - 1, col))
            out.append('<text x="%.1f" y="%d">%.0f%%</text>' % (lab + v * scale + 4, y + j * bar_h + bar_h - 1, 100 * v))
    for t in (0, 0.5, 1):
        out.append('<text x="%.1f" y="%d">%d%%</text>' % (lab + t * scale - 8, h - 2, 100 * t))
    out.append("</svg>")
    return "".join(out)


def meter(value: float, total: float | None, width: int = 520) -> str:
    if not total:
        return '<p class="dim">budget unknown (no "start:" line in the training log)</p>'
    frac = min(1.0, value / total)
    col = "var(--err)" if frac >= 1 else ("var(--warn)" if frac > 0.85 else "var(--ok)")
    return ('<svg viewBox="0 0 %d 22" width="%d" style="max-width:100%%" role="img" aria-label="spend vs budget">'
            '<rect x="0" y="4" width="%d" height="12" fill="#0b0e12" stroke="var(--line)" rx="3"/>'
            '<rect x="0" y="4" width="%.1f" height="12" fill="%s" rx="3"/></svg>') % (width, width, width, frac * width, col)


def columns(vals: list[tuple[str, float | None, str]], width: int = 520, height: int = 90, ref: float | None = 0.5,
            fmt=lambda v: "%.2f" % v) -> str:
    """Vertical bars; v is a 0..1 height, fmt(v) is the label above the bar."""
    if not vals:
        return '<p class="dim">no data</p>'
    n = len(vals)
    w = width / n
    out = ['<svg viewBox="0 0 %d %d" width="%d" style="max-width:100%%" role="img">' % (width, height + 16, width)]
    if ref is not None:
        y = height - ref * height
        out.append('<line x1="0" x2="%d" y1="%.1f" y2="%.1f" stroke="var(--line)" stroke-dasharray="4 3"/>' % (width, y, y))
    for i, (label, v, col) in enumerate(vals):
        if v is not None:
            bh = max(1, v * height)
            out.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" rx="2"/>'
                       % (i * w + w * 0.2, height - bh, w * 0.6, bh, col))
            out.append('<text x="%.1f" y="%.1f" text-anchor="middle">%s</text>' % (i * w + w / 2, height - bh - 3, e(fmt(v))))
        out.append('<text x="%.1f" y="%d" text-anchor="middle">%s</text>' % (i * w + w / 2, height + 13, e(label)))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------- sections

def sec_health(st: dict) -> str:
    rows = "".join("<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (pill(h["level"]), e(h["name"]), e(h["text"]))
                   for h in st["health"])
    return '<section class="wide"><h2>Health</h2><table><tr><th>level</th><th>check</th><th>detail</th></tr>%s</table></section>' % rows


def sec_agents(st: dict) -> str:
    parts = ['<section class="wide"><h2>Agents: held-out base vs pack</h2>',
             '<div class="legend"><span><i class="sw" style="background:var(--base)"></i>base</span>'
             '<span><i class="sw" style="background:var(--pack)"></i>with pack</span>'
             '<span class="dim">pooled p = exact McNemar over all held-out rounds; promotion needs p &lt; 0.05</span></div>']
    for b in st["training"]["batches"]:
        agents = sorted(b["agents"], key=lambda a: (a["field"], a["agent"] or ""))
        parts.append("<h3>%s</h3>" % e(b["name"]))
        parts.append(bars_pair([("%s / %s" % (a["agent"], a["field"]), rate(a["held_base"]), rate(a["held_with_pack"]))
                                for a in agents if a["held_base"]]))
        rows = []
        for a in agents:
            pk = b["packs"].get((a["agent"] or "").replace(":", "_") + "/" + a["field"], {})
            po = a["pooled"] or {}
            rows.append('<tr><td>%s</td><td>%s</td><td class="n">%s</td><td class="n">%s</td><td class="n">%s</td>'
                        '<td class="n">%s</td><td class="n">%s</td><td>%s</td><td class="n">%s</td></tr>' % (
                            e(a["agent"]), e(a["field"]), fmt_rate(a["held_base"]), fmt_rate(a["held_with_pack"]),
                            "%s/%s vs %s" % (po.get("base", "-"), po.get("n", "-"), po.get("pack", "-")) if po else "-",
                            "%s / %s" % (po.get("c_pack_only", "-"), po.get("b_base_only", "-")) if po else "-",
                            fmt_p(po.get("p")), pill(status_level(a["status"]), a["status"] or "none"),
                            "v%s, %s/%s kept" % (pk.get("version", a["pack_version"]), pk.get("kept", "-"),
                                                 pk.get("rounds", "-")) if pk else e(a["pack_version"])))
        parts.append('<table><tr><th>agent</th><th>field</th><th class="n">held base</th><th class="n">held pack</th>'
                     '<th class="n">pooled base vs pack</th><th class="n">pack-only / base-only</th><th class="n">p</th>'
                     '<th>status</th><th class="n">pack</th></tr>%s</table>' % "".join(rows))
    if not st["training"]["batches"]:
        parts.append('<p class="dim">no training batches found under %s</p>' % e(st["training"]["root"]))
    parts.append("</section>")
    return "".join(parts)


def sec_training(st: dict) -> str:
    parts = ['<section><h2>Training progress and spend</h2>']
    tot_spent = sum(b["spent"] or 0 for b in st["training"]["batches"])
    tot_budget = sum(b["budget"] or 0 for b in st["training"]["batches"])
    parts.append('<div class="kpis"><div class="kpi"><b>$%.2f</b><span>spent, all batches</span></div>'
                 '<div class="kpi"><b>%s</b><span>budget (latest run per batch)</span></div></div>'
                 % (tot_spent, "$%.0f" % tot_budget if tot_budget else "-"))
    for b in st["training"]["batches"]:
        statuses: dict[str, int] = {}
        for p in b["packs"].values():
            statuses[p["status"] or "none"] = statuses.get(p["status"] or "none", 0) + 1
        parts.append("<h3>%s: $%.2f of %s</h3>" % (e(b["name"]), b["spent"] or 0,
                                                    "$%.0f" % b["budget"] if b["budget"] else "unknown budget"))
        parts.append(meter(b["spent"] or 0, b["budget"]))
        parts.append('<table><tr><td>result rows</td><td class="n">%d (%d held-out, %d with error)</td></tr>'
                     '<tr><td>results updated</td><td class="n">%s (%s ago)</td></tr>'
                     '<tr><td>packs by status</td><td class="n">%s</td></tr>'
                     '<tr><td>quota stops</td><td class="n">%d</td></tr>'
                     '<tr><td>last log line</td><td class="n">%s</td></tr></table>' % (
                         b["results"], b["held_rows"], b["result_errors"], e(b["results_mtime"]),
                         age(b["results_age_h"]), e(", ".join("%s %d" % kv for kv in sorted(statuses.items())) or "-"),
                         b["quota_stops"], e((b["last_log"] or "-")[:140])))
    parts.append("</section>")
    return "".join(parts)


def sec_promotions(st: dict) -> str:
    promos = st["mesh"]["promotions"]
    parts = ['<section><h2>Promotions</h2>']
    if not promos:
        parts.append('<p class="dim">promotions.jsonl is empty or absent: no pack has passed the gate yet '
                     '(pooled McNemar p &lt; 0.05).</p>')
    else:
        rows = "".join('<tr><td>%s</td><td>%s</td><td>%s</td><td class="n">v%s</td><td class="n">%s</td><td>%s</td></tr>' % (
            e(p.get("t")), e(p.get("agent")), e(p.get("field")), e(p.get("version")),
            fmt_p((p.get("pooled") or {}).get("p")),
            pill("ok" if p.get("result") == "published" else "err", p.get("result")) + (
                " " + e(", ".join(p.get("why") or [])) if p.get("why") else "")) for p in promos[-12:][::-1])
        parts.append('<table><tr><th>time</th><th>agent</th><th>field</th><th class="n">pack</th><th class="n">p</th>'
                     '<th>result</th></tr>%s</table>' % rows)
    nightly_promos = [r for r in st["nightly"]["runs"] if r["promoted"]]
    if nightly_promos:
        parts.append("<h3>nightly LLM promotions (last 7 runs)</h3><table>%s</table>" % "".join(
            "<tr><td>%s</td><td>%s</td></tr>" % (e(r["ended"]), e(r["ver"])) for r in nightly_promos))
    parts.append("</section>")
    return "".join(parts)


def sec_router(st: dict) -> str:
    rt = st["mesh"]["router"]
    s = rt["state"] or {}
    fed_train = sum(b["fed_router"] for b in st["training"]["batches"])
    return ('<section><h2>Router</h2><div class="kpis">'
            '<div class="kpi"><b>%d</b><span>outcomes in outcomes.jsonl</span></div>'
            '<div class="kpi"><b>%d</b><span>fed by ingest runs</span></div>'
            '<div class="kpi"><b>%d</b><span>fed by training export</span></div></div>'
            '<table><tr><td>router_state.json</td><td class="n">%s, %s</td></tr>'
            '<tr><td>last update</td><td class="n">%s (%s ago) %s</td></tr>'
            '<tr><td>last outcome</td><td class="n">%s (%s)</td></tr></table></section>') % (
        rt["outcomes"], rt["fed_by_ingest"], fed_train, size(s.get("size")), e(s.get("mtime_iso") or "missing"),
        e(s.get("mtime_iso")), age(rt["age_h"]), pill("err", "stale") if rt["stale"] else pill("ok", "fresh"),
        e(rt["last_outcome"]), e(rt["last_outcome_project"]))


def sec_memory(st: dict) -> str:
    ex, ws = st["mesh"]["experience"], st["mesh"]["workspace"]
    parts = ['<section><h2>Experience memory and workspace</h2><div class="kpis">'
             '<div class="kpi"><b>%d</b><span>verified solutions (experience.jsonl, %s)</span></div>'
             '<div class="kpi"><b>%s</b><span>workspace entries</span></div>'
             '<div class="kpi"><b>%s</b><span>verified workspace entries</span></div></div>' % (
                 ex["n"], size((ex["stat"] or {}).get("size")),
                 ws["entries"] if ws["exists"] else "-", ws["verified"] if ws["exists"] else "-")]
    if ex["fields"]:
        top = max(ex["fields"].values())
        parts.append(columns([(k, v / top, "var(--accent)") for k, v in sorted(ex["fields"].items())], ref=None,
                             height=60, fmt=lambda v, top=top: "%d" % round(v * top)))
        parts.append('<p class="dim">%s</p>' % e(", ".join("%s %d" % kv for kv in sorted(ex["fields"].items()))))
    if ex["recent"]:
        parts.append("<h3>latest experience rows</h3><table>%s</table>" % "".join(
            "<tr><td>%s</td><td>%s</td><td>%s</td></tr>" % (e(r["task"]), e(r["agent"]), e(r["field"]))
            for r in ex["recent"]))
    if not ws["exists"]:
        parts.append('<p class="dim">workspace.db not present yet.</p>')
    else:
        parts.append('<p class="dim">workspace.db %s, kinds: %s</p>' % (
            size((ws["stat"] or {}).get("size")), e(", ".join("%s %d" % kv for kv in sorted(ws["kinds"].items())))))
        if ws["recent"]:
            parts.append('<h3>recent workspace posts</h3><table><tr><th>id</th><th>time</th><th>task</th><th>agent</th>'
                         '<th>kind</th><th></th><th>content</th></tr>%s</table>' % "".join(
                             '<tr><td class="n">%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
                                 r["id"], e(r["created"]), e(r["task"]), e(r["agent"]), e(r["kind"]),
                                 pill("ok", "verified") if r["verified"] else "", e(r["content"]))
                             for r in ws["recent"]))
    parts.append("</section>")
    return "".join(parts)


def sec_nightly(st: dict) -> str:
    ni = st["nightly"]
    prod = ni["production"] or {}
    parts = ['<section class="wide"><h2>Nightly LLM</h2><div class="kpis">'
             '<div class="kpi"><b>%s</b><span>production (%s)</span></div>'
             '<div class="kpi"><b>%s</b><span>production since</span></div>'
             '<div class="kpi"><b>%d</b><span>SFT rows trained</span></div>'
             '<div class="kpi"><b>%s</b><span>nightly.lock</span></div></div>' % (
                 e(prod.get("ver") or "-"), e(prod.get("base") or "?"), e(prod.get("since") or "-"), ni["trained"],
                 e("pid %s since %s" % (ni["lock"]["pid"], ni["lock"]["since"])) if ni["lock"] else "free")]
    runs = ni["runs"]
    parts.append(columns([((r["run_id"] or "")[4:8], r["rate"], "var(--ok)" if r["promoted"] else "var(--base)")
                          for r in runs[::-1]], height=70))
    parts.append('<p class="dim">bar = pairwise win rate (or its CI lower bound when the gate passed); dashed line = 0.5; '
                 'green = promoted.</p>')
    rows = []
    for r in runs:
        lvl = "ok" if r["exit"] == 0 and r["status"] == "ok" else "err"
        stages = " ".join("%s:%s" % (s[0], s[1]) for s in r["stages"])
        rows.append('<tr><td>%s</td><td>%s</td><td class="n">%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (
            e(r["run_id"]), pill(lvl, "%s / exit %s" % (r["status"], r["exit"])), e(r["n_new"]), e(stages),
            pill("ok", "promoted") if r["promoted"] else pill("dim", "kept prod"), e(r["gate"]),
            "synthetic" if r["synthetic"] else ""))
    parts.append('<table><tr><th>run</th><th>result</th><th class="n">new rows</th><th>stages</th><th>gate</th>'
                 '<th>gate reason</th><th></th></tr>%s</table>' % "".join(rows))
    if ni["log_tail"]:
        parts.append("<h3>nightly.log tail (%s)</h3><pre>%s</pre>" % (
            e((ni["log_stat"] or {}).get("mtime_iso")), e("\n".join(ni["log_tail"]))))
    parts.append("</section>")
    return "".join(parts)


def sec_ingest(st: dict) -> str:
    ig = st["mesh"]["ingest"]
    tot = ig["totals"] or {}
    judge = ig["judge"] or {}
    return ('<section><h2>Ingest</h2><table>'
            '<tr><td>runs recorded</td><td class="n">%d</td></tr>'
            '<tr><td>last run</td><td class="n">%s (%s ago, %ss)</td></tr>'
            '<tr><td>last exit (ingest-runs.log)</td><td class="n">%s at %s</td></tr>'
            '<tr><td>totals</td><td class="n">%s</td></tr>'
            '<tr><td>judge</td><td class="n">%s</td></tr></table></section>') % (
        ig["runs"], e(ig["last_ts"]), age(ig["age_h"]), e(ig["runtime_s"]), e(ig["last_exit"]), e(ig["last_exit_ts"]),
        e(", ".join("%s %s" % kv for kv in tot.items())), e(", ".join("%s %s" % kv for kv in judge.items())))


def render(st: dict) -> str:
    worst = st["health"][0]["level"] if st["health"] else "ok"
    body = "".join([sec_health(st), sec_agents(st), sec_training(st), sec_promotions(st), sec_router(st),
                    sec_memory(st), sec_ingest(st), sec_nightly(st)])
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta http-equiv="refresh" content="300"><title>neural-mesh ops</title><style>%s</style></head><body>'
            '<header><h1>neural-mesh ops</h1>%s<span class="dim">generated %s in %.2fs; reloads every 5 min; '
            'regenerated by the neural-mesh-dashboard task</span></header><main>%s</main></body></html>') % (
        CSS, pill(worst, {"ok": "all ok", "warn": "warnings", "err": "attention"}[worst]), e(st["generated"]),
        st["collect_s"], body)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def status_doc(st: dict) -> dict:
    """status.json: a compact machine-readable summary (no log text, no workspace content)."""
    me, ni = st["mesh"], st["nightly"]
    return {
        "generated": st["generated"], "collect_s": st["collect_s"],
        "health": st["health"],
        "worst": st["health"][0]["level"] if st["health"] else "ok",
        "training": [{"batch": b["name"], "spent": b["spent"], "budget": b["budget"], "results": b["results"],
                      "result_errors": b["result_errors"], "quota_stops": b["quota_stops"],
                      "agents": [{"agent": a["agent"], "field": a["field"], "held_base": a["held_base"],
                                  "held_with_pack": a["held_with_pack"], "status": a["status"],
                                  "pooled_p": (a["pooled"] or {}).get("p")} for a in b["agents"]]}
                     for b in st["training"]["batches"]],
        "promotions": len(me["promotions"]),
        "router": {"outcomes": me["router"]["outcomes"], "age_h": me["router"]["age_h"],
                   "stale": me["router"]["stale"], "fed_by_ingest": me["router"]["fed_by_ingest"]},
        "experience": me["experience"]["n"],
        "workspace": {"exists": me["workspace"]["exists"], "entries": me["workspace"]["entries"],
                      "verified": me["workspace"]["verified"]},
        "ingest": {k: me["ingest"][k] for k in ("last_ts", "age_h", "last_exit", "runs")},
        "nightly": {"production": (ni["production"] or {}).get("ver"), "running": ni["lock"] is not None,
                    "runs": [{k: r[k] for k in ("run_id", "status", "exit", "promoted", "rate")} for r in ni["runs"]]},
    }


def build(over: dict | None = None) -> dict:
    st = sources.collect(over)
    out = sources.roots(over)["out"]
    _atomic_write(out / "index.html", render(st))
    _atomic_write(out / "status.json", json.dumps(status_doc(st), indent=1))
    st["out"] = str(out)
    return st


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out")
    p.add_argument("--nm-root")
    p.add_argument("--ac-root")
    a = p.parse_args(argv)
    over = {k: v for k, v in (("out", a.out), ("nm", a.nm_root), ("ac", a.ac_root)) if v}
    t0 = time.time()
    st = build(over)
    print("dashboard: %s\\index.html (%.2fs, worst=%s)" % (st["out"], time.time() - t0,
                                                         st["health"][0]["level"] if st["health"] else "ok"))
    return 0


if __name__ == "__main__":
    sys.exit(main())