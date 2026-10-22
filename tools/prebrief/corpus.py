"""Build the timestamped prebrief corpus (read-only): past agent runs from transcripts, hive findings sections, journal entries.

Each record: {id, kind: run|finding|journal, ts (ISO, UTC), session, project, agent, title, text (what retrieval sees),
report (final report, runs only), metrics (runs only)}. Runs come from <project>/<session>/subagents/agent-*.jsonl + .meta.json:
first user message = task prompt, last substantial assistant text (or SendMessage body) = final report.
"""
import glob
import json
import os
import re
from datetime import datetime, timezone

from .index import anchors
from .scrub import scrub

PROJECTS = os.path.join(os.path.expanduser("~"), ".claude", "projects")
HIVE = os.path.join(os.path.expanduser("~"), "ruv", "hive")


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content or [] if isinstance(c, dict) and c.get("type") == "text")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def read_run(path: str) -> dict | None:
    meta_path = path[:-6] + ".meta.json"
    meta = {}
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    prompt, texts, first_ts, last_ts = None, [], None, None
    trace = []  # tool inputs (paths, commands) + assistant text: what the agent touched
    fresh = out = cache_read = turns = tools = 0
    for line in open(path, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        ts = o.get("timestamp")
        if ts:
            first_ts = first_ts or ts
            last_ts = ts
        m = o.get("message") or {}
        if o.get("type") == "user" and prompt is None:
            t = _text_of(m.get("content"))
            if t.strip():
                prompt = t
        elif o.get("type") == "assistant":
            turns += 1
            u = m.get("usage") or {}
            fresh += int(u.get("input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
            out += int(u.get("output_tokens") or 0)
            cache_read += int(u.get("cache_read_input_tokens") or 0)
            for c in m.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools += 1
                    inp = c.get("input") or {}
                    trace.extend(str(inp.get(k)) for k in ("file_path", "path", "command", "pattern", "notebook_path") if inp.get(k))
                    if c.get("name") == "SendMessage":  # teammates report through SendMessage
                        msg = (c.get("input") or {}).get("message")
                        if isinstance(msg, str) and len(msg) > 200:
                            texts.append(msg)
            t = _text_of(m.get("content"))
            if t.strip():
                texts.append(t)
                trace.append(t)
    if not prompt or not first_ts:
        return None
    long_texts = [t for t in texts if len(t) > 200]
    report = long_texts[-1] if long_texts else "\n".join(texts[-3:])
    parts = path.replace("\\", "/").split("/")
    project, session = parts[-4], parts[-3]
    agent_id = os.path.basename(path)[6:-6]
    # scrub FULL texts before any truncation: a secret cut at a boundary no longer matches and would leak partially
    prompt, report = scrub(prompt), scrub(report)
    desc = scrub(meta.get("description") or prompt.strip().splitlines()[0])[:100]
    t0 = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
    return {
        "id": f"run:{session[:8]}:{agent_id}", "kind": "run", "ts": _iso(t0), "ts_end": _iso(t1),
        "session": session, "project": project, "agent": meta.get("agentType") or "unknown",
        "title": desc, "prompt": prompt[:6000], "report": report[:8000],
        "text": f"{desc}\n{prompt[:3000]}\n{report[:4000]}",
        "trace_anchors": sorted(anchors(scrub("\n".join(trace)))),
        "metrics": {"fresh_in": fresh, "out": out, "cache_read": cache_read, "turns": turns, "tools": tools,
                    "wall_s": round((t1 - t0).total_seconds(), 1)},
    }


def runs(root: str = PROJECTS) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "subagents", "agent-*.jsonl"))):
        r = read_run(p)
        if r:
            out.append(r)
    return out


_HEAD = re.compile(r"^#{1,3} ", re.M)


def findings(hive: str = HIVE) -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(hive, "findings", "*.md"))):
        name = os.path.basename(p)[:-3]
        body = open(p, encoding="utf-8-sig", errors="replace").read()
        ts = _iso(datetime.fromtimestamp(os.path.getmtime(p), timezone.utc))
        starts = [m.start() for m in _HEAD.finditer(body)]
        if not starts or starts[0] != 0:
            starts = [0] + starts
        for i, s in enumerate(starts):
            sec = scrub(body[s: starts[i + 1] if i + 1 < len(starts) else len(body)].strip())
            if len(sec) < 80:
                continue
            title = sec.splitlines()[0].lstrip("# ")[:120]
            out.append({"id": f"finding:{name}:{i}", "kind": "finding", "ts": ts, "session": "hive", "project": "hive",
                        "agent": name, "title": title, "text": sec[:5000], "source": f"findings/{name}.md"})
    return out


def journal(hive: str = HIVE) -> list[dict]:
    out = []
    p = os.path.join(hive, "journal.jsonl")
    if not os.path.exists(p):
        return out
    for i, line in enumerate(open(p, encoding="utf-8-sig", errors="replace")):
        try:
            o = json.loads(line)
            ts = _iso(datetime.fromisoformat(o["ts"]).astimezone())  # log.ps1 writes local time without offset
        except (ValueError, KeyError):
            continue
        txt = scrub(o.get("text", ""))
        out.append({"id": f"journal:{i}", "kind": "journal", "ts": ts, "session": o.get("session", ""), "project": "hive",
                    "agent": o.get("agent", ""), "title": f"{o.get('kind', '')}: {txt[:100]}",
                    "text": f"{o.get('kind', '')} {txt} {scrub(o.get('ref', ''))}", "source": "journal.jsonl"})
    return out


def build(out_path: str) -> list[dict]:
    docs = runs() + findings() + journal()
    docs.sort(key=lambda d: d["ts"])
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return docs


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]
