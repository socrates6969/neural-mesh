"""prebrief(task_text, agent_role) -> markdown <= 600 tokens (chars/4), built by the queen BEFORE dispatch and prepended to the prompt.

CLI: python -m prebrief "<task text>" [--role <agent type>] [--session <id to exclude>] [--repo <codemem repo>] [--max-tokens 600] [--json]
     python -m prebrief --rebuild            # refresh corpus.jsonl from transcripts + hive findings + journal
Retriever = BM25 over the corpus (offline O1/E1: dense fusion did not beat BM25, findings/prebrief.md).
"""
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from . import corpus as corpus_mod
from .index import NM, Index, anchors, tokens
from .scrub import scrub

RUNS = os.path.join(os.path.expanduser("~"), "ruv", "hive", "runs", "prebrief")
CORPUS = os.path.join(RUNS, "corpus.jsonl")
CACHE = os.path.join(RUNS, "index.pkl")
LOG = os.path.join(RUNS, "brief_log.jsonl")
CODEMEM = os.path.join(NM, "target", "release", "codemem.exe")
HM_DIR = os.path.join(os.path.expanduser("~"), "ruv", "hive", "wt", "hive-memory")
HM_PY = os.path.join(os.path.expanduser("~"), "ruv", "venvs", "hive-memory", "Scripts", "python.exe")

NEG = re.compile(r"\b(reject\w*|fail\w*|worse|regress\w*|bug|broke\w*|pitfall|do not|don't|never|not significant|no gain|"
                 r"timeout|crash\w*|leak\w*|wrong|blocked|killed|0/\d+|abandon\w*|lost|stale|underpowered|artifact|artefact)\b", re.I)
SENT = re.compile(r"(?<=[.!?;])\s+|\n+")
REPOS = [("hebbian-router", ("hebbian-router",)), ("freight-recovery-build", ("freight", "scoup", "carrier", "invoice", "tariff", "vitest")),
         ("neural-mesh", ("neural-mesh", "codemem", "hebbian", "router_state", "ingest/"))]


def load_index(rebuild: bool = False) -> Index:
    if rebuild or not os.path.exists(CORPUS):
        os.makedirs(RUNS, exist_ok=True)
        corpus_mod.build(CORPUS)
    m = os.path.getmtime(CORPUS)
    if os.path.exists(CACHE):
        try:
            with open(CACHE, "rb") as f:
                cm, idx = pickle.load(f)
            if cm == m:
                return idx
        except Exception:
            pass
    idx = Index(corpus_mod.load(CORPUS))
    with open(CACHE, "wb") as f:
        pickle.dump((m, idx), f)
    return idx


def _sentences(text: str) -> list:
    out = []
    for s in SENT.split(text):
        s = re.sub(r"\s+", " ", s.strip(" -*#|>"))
        if 25 <= len(s) <= 320:
            out.append(s)
    return out


def _best(text: str, qset: set, k: int = 2, need=None) -> list:
    scored = []
    for i, s in enumerate(_sentences(text)):
        if need is not None and not need.search(s):
            continue
        st = set(tokens(s))
        ov = len(st & qset)
        bonus = 0.5 * bool(re.search(r"\d", s)) + 0.5 * bool(anchors(s))
        if ov:
            scored.append((ov + bonus, -i, s))
    scored.sort(reverse=True)
    return [s for _, _, s in scored[:k]]


def _prov(d: dict) -> str:
    day = d["ts"][:10]
    if d["kind"] == "run":
        return f"{day} {d['agent']} run {d['session'][:8]}"
    return f"{day} {d.get('source', d['kind'])}" + (f" ({d['agent']})" if d["kind"] == "journal" else "")


def _codemem(task: str, repo: str | None, k: int = 4) -> list:
    """BM25 code pointers; without --repo query the named repo or both main repos and merge by score."""
    if not os.path.exists(CODEMEM):
        return []
    low = task.lower()
    repos = [repo] if repo else [r for r, keys in REPOS if keys and any(x in low for x in keys)] or ["neural-mesh", "freight-recovery-build"]
    hits = []
    for rp in repos:
        try:
            r = subprocess.run([CODEMEM, "query", task[:600], "--repo", rp, "--k", str(k), "--json"], capture_output=True, text=True, timeout=10)
            out = json.loads(r.stdout) if r.returncode == 0 else {}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
        cov = out.get("coverage") or 0
        hits += [(p["score"] * cov, f"{rp}/{p['path']}:{p['line']} ({p.get('title', '')})") for p in out.get("pointers", [])]
    hits.sort(reverse=True)
    return [h for _, h in hits[:k]]


def _hive_memory(task: str, session: str | None, k: int = 3) -> list:
    """Opt-in: verified prior work from hive-memory's corpus API (schema hive-corpus-query/1). Only exec-verified outcomes are shown."""
    if not (os.path.exists(HM_PY) and os.path.isdir(HM_DIR)):
        return []
    cmd = [HM_PY, "-m", "ingest.corpus_index", "query", "-", "-k", "8", "--patterns", "3", "--json"]
    if session:
        cmd += ["--exclude-sessions", session]
    try:
        r = subprocess.run(cmd, input=task[:3000], capture_output=True, text=True, timeout=30, cwd=HM_DIR, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"})
        out = json.loads(r.stdout) if r.returncode == 0 else {}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return []
    lines = []
    for h in out.get("hits", []):
        if h.get("outcome") in ("verified_pass", "commit_landed"):
            snip = re.sub(r"\s+", " ", h.get("snippet", ""))[:220]
            lines.append(f"- [{h['ts'][:10]} {h.get('agent_type') or h.get('agent')} {h['outcome']} {h['session'][:8]}] {snip}")
    for pt in out.get("patterns", []):
        files = ", ".join(os.path.basename(f) for f in (pt.get("files") or [])[:3])
        if files:
            lines.append(f"- [{pt['ts'][:10]} {pt.get('agent_type')} {pt.get('outcome')} pattern] files: {files}")
    return lines[:k]


def prebrief(task: str, role: str | None = None, session: str | None = None, repo: str | None = None,
             max_tokens: int = 600, k: int = 10, idx: Index | None = None, source: str = "local", before: str | None = None) -> dict:
    t0 = time.time()
    idx = idx or load_index()
    avail = idx.available(before, session)  # before = ISO ts upper bound (leak-free briefs)
    q = f"{task}\n{role or ''}"
    s = idx.search(q, role, avail, method="B")
    order = [int(i) for i in np.argsort(-s, kind="stable")[:k] if np.isfinite(s[i]) and s[i] > 0]
    qset = set(tokens(task))
    docs = [idx.docs[i] for i in order]

    sol = []
    for d in docs:
        body = d.get("report") or d["text"]
        best = _best(body, qset, 2)
        if best:
            sol.append(f"- [{_prov(d)}] {d['title'][:90]}: " + " ".join(best))
        if len(sol) == 3:
            break
    pit, seen = [], set()
    for d in docs:
        for b in _best(d.get("report") or d["text"], qset, 1, need=NEG):
            if b not in seen:
                seen.add(b)
                pit.append(f"- [{_prov(d)}] {b}")
        if len(pit) == 3:
            break
    paths = Counter()
    for d in docs[:5]:
        for a in anchors(d.get("report") or d["text"]):
            if a.startswith("p:"):
                paths[a[2:]] += 1
    files = _codemem(task, repo) + [f"{p} (in {n} similar runs)" for p, n in paths.most_common(4) if n >= 2]
    votes = Counter()
    for rank, d in enumerate(docs):
        if d["kind"] == "run" and d["agent"] not in ("unknown", ""):
            votes[d["agent"]] += 1.0 / (1 + rank)
    agent_line = ""
    if votes:
        a, w = votes.most_common(1)[0]
        n = sum(1 for d in docs if d.get("agent") == a)
        agent_line = f"- similar past tasks used `{a}` ({n} of top {len(docs)}; hint only, not a router)"

    head = "## Pre-brief (auto, prebrief v1; prior hive work, verify before relying on it)"
    verified = _hive_memory(task, session) if source == "hive-memory" else []
    sections = [("### Verified prior work (hive-memory)", verified), ("### Prior solutions", sol), ("### Known pitfalls", pit), ("### Relevant files", [f"- {x}" for x in files[:6]]),
                ("### Agent hint", [agent_line] if agent_line else [])]
    budget = max_tokens * 4
    out = [head]
    used = len(head)
    for title, lines in sections:
        if not lines:
            continue
        block = [title]
        for ln in lines:
            ln = ln if len(ln) <= 420 else ln[:417] + "..."
            if used + sum(len(x) + 1 for x in block) + len(ln) + 1 > budget:
                break
            block.append(ln)
        if len(block) > 1:
            out.extend(block)
            used += sum(len(x) + 1 for x in block)
    md = scrub("\n".join(out)) if docs else ""
    res = {"markdown": md, "tokens": len(md) // 4, "doc_ids": [d["id"] for d in docs[:5]], "ms": round((time.time() - t0) * 1000, 1)}
    try:
        if os.environ.get("PREBRIEF_LOG") == "off":
            raise OSError("log off")
        os.makedirs(RUNS, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "task_sha": hashlib.sha1(task.encode("utf-8")).hexdigest()[:12], "role": role,
                                "arm": "brief", "source": source, "tokens": res["tokens"], "doc_ids": res["doc_ids"], "ms": res["ms"]}) + "\n")
    except OSError:
        pass
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="prebrief")
    ap.add_argument("task", nargs="?")
    ap.add_argument("--role")
    ap.add_argument("--session")
    ap.add_argument("--repo")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--task-file")
    ap.add_argument("--source", default="local", choices=["local", "hive-memory"])
    ap.add_argument("--before", help="only documents with ts < this ISO UTC timestamp")
    a = ap.parse_args(argv)
    if a.rebuild:
        idx = load_index(rebuild=True)
        print(f"corpus: {len(idx.docs)} docs -> {CORPUS}")
        if not (a.task or a.task_file):
            return 0
    task = open(a.task_file, encoding="utf-8").read() if a.task_file else a.task
    if not task:
        ap.error("task text required")
    r = prebrief(task, a.role, a.session, a.repo, a.max_tokens, source=a.source, before=a.before)
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1) if a.json else r["markdown"])
    return 0