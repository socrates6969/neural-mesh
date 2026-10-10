"""(ii) agent-task A/B: does codemem make `claude -p` (haiku, read-only tools) find the right file:line faster / cheaper?
Arms (same question, same read-only tool set incl. read-only Bash search, same answer-format instruction in --append-system-prompt):
  A  baseline
  B  the REAL UserPromptSubmit hook (hooks/settings-hooks-codemem.json passed via --settings) -> pointers injected only when it fires
  C  CLAUDE.md-style snippet (bench/codemem/snippet.md) + `codemem` allowed in Bash, no hook
Per run: stream-json is parsed for tool calls (Read/Grep/Glob, search-like Bash = grep/rg/find/cat/ls/head/tail/sed -n/git grep, codemem calls),
usage (fresh = input + cache_creation + output; all = + cache_read), turns, duration, cost; the answer is graded against the key
(file must match; line within the key range +-5).
  python agent_ab.py run --tag t1 [--split held|tune|all] [--arms A,B,C] [--reps 2] [--workers 4] [--limit N]
  python agent_ab.py report --tag t1 [--split held]
Resumable: finished (id, arm, rep) rows in results/<tag>/raw.jsonl are skipped."""
import argparse, concurrent.futures as cf, json, math, os, random, re, statistics, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NM = HERE.parents[1]
EXE_DIR = NM / "data" / "codemem" / "bin"
SETTINGS_B = NM / "hooks" / "settings-hooks-codemem.json"
SNIPPET = (HERE / "snippet.md").read_text(encoding="utf8")
FORMAT = ("You are answering a question about the code in the current working directory. Find the single most relevant location. "
          "Finish with exactly one line `ANSWER: <path relative to the repo root>:<line number>` (forward slashes). Be efficient.")
READ_ONLY = ["Read", "Grep", "Glob"] + [f"Bash({c}:*)" for c in ("grep", "rg", "find", "cat", "ls", "head", "tail", "wc", "sed", "git grep", "git ls-files")]
SEARCHY = re.compile(r"(^|[\s;|&(])(grep|rg|find|cat|ls|head|tail|sed|git\s+grep|git\s+ls-files|wc)\b")


def run_claude(q, arm, timeout=420):
    env = dict(os.environ, PATH=str(EXE_DIR) + os.pathsep + os.environ.get("PATH", ""), NEURAL_MESH_CODEMEM_LOG="off")
    env.pop("NEURAL_MESH_CODEMEM", None)
    sysp = FORMAT + ("\n\n" + SNIPPET if arm == "C" else "")
    tools = READ_ONLY + (["Bash(codemem:*)"] if arm == "C" else [])
    cmd = ["claude", "-p", q["question"], "--model", "haiku", "--output-format", "stream-json", "--verbose", "--setting-sources", "project",
           "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence", "--max-turns", "30", "--append-system-prompt", sysp, "--allowedTools", *tools]
    if arm == "B":
        cmd += ["--settings", str(SETTINGS_B)]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=q["cwd"], env=env, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=timeout, stdin=subprocess.DEVNULL)
        out, rc = r.stdout, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or ""), -9
    return out, rc, (time.perf_counter() - t0) * 1000


def parse(out):
    calls = {"read": 0, "grep_glob": 0, "bash_search": 0, "bash_other": 0, "codemem": 0, "other": 0}
    res, text, hook_ctx = None, "", False
    for line in out.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "assistant":
            for c in d["message"]["content"]:
                if c.get("type") == "text":
                    text = c["text"]
                elif c.get("type") == "tool_use":
                    n, i = c["name"], c.get("input", {})
                    if n == "Read": calls["read"] += 1
                    elif n in ("Grep", "Glob"): calls["grep_glob"] += 1
                    elif n == "Bash":
                        cmd = i.get("command", "")
                        if re.search(r"\bcodemem(\.exe)?\b", cmd): calls["codemem"] += 1
                        elif SEARCHY.search(cmd): calls["bash_search"] += 1
                        else: calls["bash_other"] += 1
                    else: calls["other"] += 1
        elif d.get("type") == "system" and "codemem" in json.dumps(d)[:4000] and d.get("subtype", "").startswith("hook"):
            hook_ctx = True
        elif d.get("type") == "result":
            res = d
    return res, text, calls, hook_ctx


def grade(text, res, q):
    blob = (res or {}).get("result") or text or ""
    m = None
    for m in re.finditer(r"ANSWER:\s*`?([^\s`:]+):(\d+)", blob):
        pass
    if not m:
        return None, None, False, False
    path = m.group(1).replace("\\", "/")
    root = q["cwd"].rstrip("/") + "/"
    if path.lower().startswith(root.lower()): path = path[len(root):]
    path = path.lstrip("./")
    line = int(m.group(2))
    file_ok = any(path == a["file"] for a in q["answers"])
    line_ok = any(path == a["file"] and a["l0"] - 5 <= line <= a["l1"] + 5 for a in q["answers"])
    return path, line, file_ok, line_ok


def one(q, arm, rep, tag):
    out, rc, wall = run_claude(q, arm)
    res, text, calls, hook_ctx = parse(out)
    u = (res or {}).get("usage") or {}
    fresh = u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("output_tokens", 0)
    path, line, file_ok, line_ok = grade(text, res, q)
    return {"id": q["id"], "arm": arm, "rep": rep, "split": q["split"], "rc": rc, "ok_run": bool(res) and not res.get("is_error"), "answer": f"{path}:{line}" if path else None,
            "file_ok": file_ok, "line_ok": line_ok, "fresh_tokens": fresh, "all_tokens": fresh + u.get("cache_read_input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
            "turns": (res or {}).get("num_turns"), "dur_ms": (res or {}).get("duration_ms"), "wall_ms": round(wall), "cost": (res or {}).get("total_cost_usd"),
            "denials": len((res or {}).get("permission_denials") or []), **calls, "tool_calls": sum(calls.values()), "hook_seen": hook_ctx}


def load_q(split):
    qs = [json.loads(l) for l in (HERE / "questions.jsonl").read_text(encoding="utf8").splitlines() if l.strip()]
    return qs if split == "all" else [q for q in qs if q["split"] == split]


def cmd_run(a):
    d = HERE / "results" / a.tag; d.mkdir(parents=True, exist_ok=True)
    raw = d / "raw.jsonl"
    done = {(r["id"], r["arm"], r["rep"]) for r in map(json.loads, raw.read_text(encoding="utf8").splitlines())} if raw.exists() else set()
    qs = load_q(a.split)[: a.limit or None]
    arms = a.arms.split(",")
    jobs = []
    for rep in range(a.reps):
        for q in qs:
            order = arms[:]; random.Random(f"{q['id']}{rep}").shuffle(order)
            jobs += [(q, arm, rep) for arm in order if (q["id"], arm, rep) not in done]
    print(len(jobs), "runs to do", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, q, arm, rep, a.tag) for q, arm, rep in jobs]
        for i, f in enumerate(cf.as_completed(futs)):
            row = f.result()
            with raw.open("a", encoding="utf8") as fh: fh.write(json.dumps(row) + "\n")
            print(i + 1, row["id"], row["arm"], row["rep"], "ok" if row["line_ok"] else ("file" if row["file_ok"] else "x"), row["fresh_tokens"], row["tool_calls"], row["dur_ms"], flush=True)


def boot_ci(diffs, n=4000, seed=7):
    rnd = random.Random(seed); m = len(diffs)
    bs = sorted(sum(rnd.choice(diffs) for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, bs[int(.025 * n)], bs[int(.975 * n)]


def cmd_report(a):
    rows = [json.loads(l) for l in (HERE / "results" / a.tag / "raw.jsonl").read_text(encoding="utf8").splitlines()]
    if a.split != "all": rows = [r for r in rows if r["split"] == a.split]
    by = {}
    for r in rows: by.setdefault((r["id"], r["arm"]), []).append(r)
    def per(item, arm, key):
        v = [float(x[key]) if not isinstance(x[key], bool) else float(x[key]) for x in by.get((item, arm), []) if x[key] is not None and x["ok_run"]]
        return statistics.mean(v) if v else None
    items = sorted({r["id"] for r in rows}); arms = sorted({r["arm"] for r in rows})
    print(f"tag={a.tag} split={a.split} items={len(items)} runs={len(rows)} failed_runs={sum(not r['ok_run'] for r in rows)}")
    keys = [("line_ok", "line acc"), ("file_ok", "file acc"), ("fresh_tokens", "fresh tok"), ("all_tokens", "all tok"), ("tool_calls", "tool calls"), ("bash_search", "bash search"),
            ("read", "Read"), ("grep_glob", "Grep/Glob"), ("codemem", "codemem"), ("turns", "turns"), ("dur_ms", "dur ms"), ("cost", "cost $")]
    print("arm means:", *[f"{arm}" for arm in arms], sep="  ")
    for k, name in keys:
        vals = []
        for arm in arms:
            v = [per(i, arm, k) for i in items]; v = [x for x in v if x is not None]
            vals.append(f"{statistics.mean(v):.3g}" if v else "-")
        print(f"  {name:12s}", *vals, sep="  ")
    base = "A"
    for arm in [x for x in arms if x != base]:
        print(f"paired {arm} - {base} (per-item means, 95% bootstrap CI):")
        for k, name in keys:
            d = [per(i, arm, k) - per(i, base, k) for i in items if per(i, arm, k) is not None and per(i, base, k) is not None]
            if len(d) >= 5:
                m, lo, hi = boot_ci(d); rel = ""
                if k in ("fresh_tokens", "all_tokens", "dur_ms", "tool_calls", "cost"):
                    ba = statistics.mean([per(i, base, k) for i in items if per(i, base, k) is not None]); rel = f"  ({m / ba * 100:+.0f}%)"
                print(f"  {name:12s} {m:+.3g} [{lo:+.3g}, {hi:+.3g}] n={len(d)}{rel}{'  *' if lo > 0 or hi < 0 else ''}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--tag", required=True); r.add_argument("--split", default="held"); r.add_argument("--arms", default="A,B,C")
    r.add_argument("--reps", type=int, default=2); r.add_argument("--workers", type=int, default=4); r.add_argument("--limit", type=int, default=0)
    p = sp.add_parser("report"); p.add_argument("--tag", required=True); p.add_argument("--split", default="held")
    a = ap.parse_args(); {"run": cmd_run, "report": cmd_report}[a.cmd](a)
