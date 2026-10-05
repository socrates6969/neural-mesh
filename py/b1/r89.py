"""R89: compute-once query-embedding MEMO hit-rate on RECORDED agent queries (the R75 'not measured: no real traffic' gap). Source: tool_use inputs in the recorded Claude Code transcripts on this box
(main + subagents): search-like inputs = Grep.pattern, Glob.pattern, WebSearch.query, any tool input field named query/pattern/q/search, memory_* tools' query, plus Agent/Task prompts.
Aggregates only (counts, rates); no text is stored or printed. Metric = share of lookups whose exact (raw) / normalised (lowercase, collapsed whitespace) string was already seen
earlier in the SAME session (per-session memo) or in ANY earlier session in time order (persistent memo), under a bounded FIFO cap (the shipped MemoEmbedding is a bounded FIFO)."""
import os, json, glob, collections, re
import numpy as np
fs = glob.glob(os.path.expanduser("~/.claude/projects/*/**/*.jsonl"), recursive=True)
KEYS = ("query", "pattern", "q", "search", "prompt")
def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
ev = collections.defaultdict(list)  # session -> [(ts, kind, text)]
for f in fs:
    p = f.replace("\\", "/"); sid = p.split("/")[-3] if "/subagents/" in p else os.path.basename(p)[:-6]
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: o = json.loads(line)
        except Exception: continue
        if o.get("type") != "assistant": continue
        ts = o.get("timestamp") or ""
        for c in o.get("message", {}).get("content", []):
            if c.get("type") != "tool_use": continue
            name = c.get("name") or ""; inp = c.get("input") or {}
            if name in ("Agent", "Task"):
                t = inp.get("prompt"); kind = "delegation"
            else:
                t = None; kind = "search"
                for k in KEYS:
                    if isinstance(inp.get(k), str) and k != "prompt": t = inp[k]; break
            if isinstance(t, str) and len(t) > 2: ev[sid].append((ts, kind, name, t))
def sim(streams, cap, key, persistent):
    hits = n = 0; mem = collections.OrderedDict()
    for st in streams:
        if not persistent: mem = collections.OrderedDict()
        for ts, kind, name, t in st:
            k = key(t); n += 1
            if k in mem: hits += 1
            else:
                mem[k] = 1
                if len(mem) > cap: mem.popitem(last=False)
    return hits, n
for kind in ("search", "delegation"):
    ses = []
    for sid, evs in ev.items():
        e = sorted([x for x in evs if x[1] == kind], key=lambda x: x[0])
        if e: ses.append(e)
    ses.sort(key=lambda e: e[0][0]); tot = sum(len(e) for e in ses)
    uniq = len({x[3] for e in ses for x in e}); print(f"\n[{kind}] sessions/agents with lookups: {len(ses)}, lookups {tot}, distinct raw strings {uniq} ({uniq/max(tot,1):.3f} of lookups)")
    toolc = collections.Counter(x[2] for e in ses for x in e); print("   by tool:", dict(toolc.most_common(6)))
    for per, persistent in (("per-session memo", False), ("persistent memo (time-ordered)", True)):
        for cap in (64, 256, 4096):
            h, n = sim(ses, cap, lambda t: t, persistent); h2, _ = sim(ses, cap, norm, persistent)
            print(f"   {per:30s} cap {cap:5d}: raw hit-rate {h/n:.3f}  normalised {h2/n:.3f}  ({n} lookups)")
    # an encoder call saved = a hit; at 3.8 ms fp32 (R85) the saved encode time per 1000 lookups = ... 
