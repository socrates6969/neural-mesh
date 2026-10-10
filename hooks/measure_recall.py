"""Measure the recall hook on REAL user prompts (read from the ingest state + raw transcripts, scrubbed; nothing is printed except aggregates
and short, scrubbed excerpts when --show is given).

  python hooks/measure_recall.py prompts [--n 300]            -> how many real prompts are available
  python hooks/measure_recall.py fire    [--n 300] [--show 8]  -> gate pass rate, fire rate, mean added tokens (in-process, no logging)
  python hooks/measure_recall.py latency [--n 50]              -> end-to-end subprocess wall time (python -S hooks/recall_hook.py), p50/p95
Tokens are estimated as chars/4 (labelled estimate). Prompts keep their own cwd (project) so the pool matches what the hook sees live.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hooks"))
import recall_hook as rh  # noqa: E402
from ingest.config import Config  # noqa: E402
from ingest.scrub import scrub  # noqa: E402


def load_prompts(n):
    """Real main-session user prompts, oldest first, de-duplicated, scrubbed, secret-flagged ones dropped."""
    cfg = Config()
    rows = {}
    for f in glob.glob(str(cfg.files_dir / "*.json")):
        try:
            d = json.load(open(f, encoding="utf8"))
        except (OSError, ValueError):
            continue
        if d.get("agent_id"):
            continue
        for t in d.get("turns", []):
            u = t.get("user") or {}
            if u.get("t") and not u.get("s"):
                rows[(d["session"], t["ts"])] = (t["ts"], d.get("cwd") or "", d["session"], u["t"])
    for f in glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")):   # transcripts not ingested yet
        try:
            for line in open(f, encoding="utf8", errors="replace"):
                o = json.loads(line)
                if o.get("type") != "user" or o.get("isMeta") or o.get("isSidechain") or o.get("isCompactSummary"):
                    continue
                c = (o.get("message") or {}).get("content")
                if isinstance(c, list):
                    if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                        continue
                    c = "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
                if not isinstance(c, str) or not c.strip() or c.lstrip().startswith(("<", "Caveat:")):
                    continue
                r = scrub(c[:4000])
                if r.secret:
                    continue
                rows.setdefault((o.get("sessionId"), o.get("timestamp")), (o.get("timestamp") or "", o.get("cwd") or "", o.get("sessionId") or "", r.text))
        except (OSError, ValueError):
            continue
    out, seen = [], set()
    for ts, cwd, sess, text in sorted(rows.values()):
        if text in seen:
            continue
        seen.add(text)
        out.append({"ts": ts, "cwd": cwd, "session": sess, "prompt": text})
    return out[-n:]


def cmd_fire(a):
    ps = load_prompts(a.n)
    if a.cwd:   # sensitivity: pretend every prompt was typed inside this directory
        ps = [dict(p, cwd=a.cwd) for p in ps]
    cfg = Config()
    gated = fired = 0
    toks = []
    shown = 0
    for p in ps:
        items_projects = None
        blk = rh.build_block(p["prompt"], p["cwd"], p["session"], cfg=cfg, log=False)
        from ingest.memory import MemoryStore
        if items_projects is None:
            st = MemoryStore(cfg)
            items_projects = {i.get("project") for i in st.items.values() if i.get("project")} | {rh.project_of_cwd(p["cwd"])}
        gated += rh.gate_score(p["prompt"], items_projects) >= rh.MIN_GATE_SCORE
        if blk:
            fired += 1
            toks.append(len(blk) / 4)
            if shown < a.show:
                shown += 1
                print("PROMPT:", p["prompt"][:160].replace("\n", " "), "\n", blk[:700], "\n---")
    n = len(ps)
    print(json.dumps({"n_prompts": n, "span": [ps[0]["ts"][:10], ps[-1]["ts"][:10]] if ps else None, "gate_pass": gated, "gate_rate": round(gated / n, 3),
                      "fired": fired, "fire_rate": round(fired / n, 3), "mean_tokens_when_fired_est": round(sum(toks) / len(toks), 1) if toks else 0,
                      "mean_tokens_per_prompt_est": round(sum(toks) / n, 1), "max_tokens_est": round(max(toks), 1) if toks else 0,
                      "min_cos": rh.MIN_COS, "token_estimate": "chars/4"}, indent=1))


def cmd_latency(a):
    import shutil
    import tempfile
    ps = load_prompts(400)
    tmp = tempfile.mkdtemp(prefix="recall-lat-")           # a private copy: the measurement must not write retrieval events into the real log
    Path(tmp, "memory").mkdir()
    shutil.copy(Config().memory_dir / "items.jsonl", Path(tmp, "memory", "items.jsonl"))
    env = dict(os.environ)
    env["INGEST_DATA"] = tmp
    if a.worst:   # worst case: every prompt passes the gate and sees the biggest pool (cwd = the neural-mesh repo)
        import recall_hook as rh2
        projs = {"neural-mesh"}
        ps = [dict(p, cwd=str(ROOT)) for p in ps if rh2.gate_score(p["prompt"], projs) >= rh2.MIN_GATE_SCORE]
        print(f"worst-case sample: {len(ps)} gate-passing prompts, cwd=neural-mesh")
    sample = ps[-a.n:]
    ts, fired = [], 0
    for p in sample:
        payload = json.dumps({"prompt": p["prompt"], "cwd": p["cwd"], "session_id": "measure-latency", "hook_event_name": "UserPromptSubmit"})
        t = time.perf_counter()
        r = subprocess.run([sys.executable, "-S", str(ROOT / "hooks" / "recall_hook.py")], input=payload, capture_output=True, text=True,
                           encoding="utf-8", env=env, timeout=10)
        ts.append(time.perf_counter() - t)
        fired += bool(r.stdout.strip())
        assert r.returncode == 0
    ts.sort()
    q = lambda x: round(ts[min(len(ts) - 1, int(x * len(ts)))] * 1000)
    print(json.dumps({"n": len(ts), "fired": fired, "p50_ms": q(0.5), "p95_ms": q(0.95), "max_ms": round(ts[-1] * 1000), "mean_ms": round(sum(ts) / len(ts) * 1000)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "fire", "latency"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--worst", action="store_true")
    ap.add_argument("--cwd")
    a = ap.parse_args()
    if a.cmd == "prompts":
        ps = load_prompts(10**6)
        print(json.dumps({"available": len(ps), "span": [ps[0]["ts"][:10], ps[-1]["ts"][:10]]}))
    elif a.cmd == "fire":
        cmd_fire(a)
    else:
        cmd_latency(a)
