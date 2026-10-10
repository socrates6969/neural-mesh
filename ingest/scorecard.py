"""Cross-session scorecard: tokens, wall time, tool calls, searching, hook overhead, waste and compaction per session and per day.

Aggregate numbers only. No prompt, response or tool text is read into any output (the per-file cache holds counts, timestamps and
tool NAMES only). Reuses the ingest code: ``reader.discover`` / ``_is_real_user`` (what a real user turn is), ``extract.parse_ts``,
``util`` I/O, and the ingest labeler's verdicts in ``outcomes.jsonl`` (extract.classify wrote them) for the re-dispatch and fix-round rates.

Baseline: ``scorecard_baseline.json`` holds the cutoff time (written once, on the first run = "everything up to the round-3 changes").
Turns and dispatches are classified by their own start time; later runs report after-minus-baseline deltas with a session-cluster
bootstrap CI (sessions are resampled, because the turns inside a session are not independent).

Known measurement limits (also printed in the report):
  * output tokens are under-recorded: transcripts keep the usage of a streamed message as written, often an early partial value;
    ``out_tiny_share`` (messages with output_tokens <= 3 that carry content) measures how common that is. Input/cache fields are reliable.
  * a leader turn counts the leader's own tokens; subagent tokens are counted per dispatch (their own transcript) and in the session total.
  * turn wall time is active time (user message to the last assistant/tool-result record) and includes permission-prompt waits.
  * hook wall time is the max duration per (toolUseID, event) group (hooks of one event run in parallel); raw summed time is kept as hook_ms_sum.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .config import Config
from .extract import parse_ts
from .reader import _is_real_user, discover
from .util import atomic_write_json, atomic_write_text, iter_jsonl, read_json, sha16

CACHE_VERSION = 3
SEARCH_TOOLS = {"Read", "Grep", "Glob"}
FAST_TOOLS = {"Read"}                                        # the no-hook floor: a Read returns in tens of ms, so any excess on the tools below is hooks, permission waits or real work
GAP_TOOLS = ("Read", "Grep", "Glob", "Edit", "Write", "Bash")
DISPATCH_TOOLS = {"Agent", "Task"}
HOOK_TYPES = {"hook_success", "hook_non_blocking_error", "hook_cancelled", "hook_blocking_error"}
GAP_CAP = 4000
_BASH_SEARCH = re.compile(r"(?:^|[;&|(]\s*)(?:grep|rg|find|fd|cat|ls|head|tail|tree|sed -n)\b")   # read-only looking commands; only the COUNT is kept
BASELINE_NOTE = "baseline = every turn/dispatch that started before this cutoff (the round-3 changes had not taken effect yet)"


# ----------------------------------------------------------------------------------------------------------------- statistics
def quantile(xs, q: float) -> float:
    """Linear-interpolation quantile (type 7); NaN for an empty list."""
    v = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return float("nan")
    if len(v) == 1:
        return float(v[0])
    pos = (len(v) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return float(v[lo] + (v[hi] - v[lo]) * (pos - lo))


def median(xs) -> float:
    return quantile(xs, 0.5)


def p90(xs) -> float:
    return quantile(xs, 0.9)


def cluster_bootstrap_delta(base: dict, post: dict, stat=median, reps: int = 1000, seed: int = 7, min_clusters: int = 3):
    """after - baseline of ``stat`` with a 95% percentile CI, resampling whole sessions inside each side.
    ``base`` / ``post`` map session -> list of unit values. Returns None when a side has fewer than ``min_clusters`` sessions."""
    b = [v for v in base.values() if v]
    p = [v for v in post.values() if v]
    if len(b) < min_clusters or len(p) < min_clusters:
        return None
    rnd = random.Random(seed)

    def draw(groups):
        return [x for g in (groups[rnd.randrange(len(groups))] for _ in groups) for x in g]

    point = stat([x for g in p for x in g]) - stat([x for g in b for x in g])
    ds = sorted(stat(draw(p)) - stat(draw(b)) for _ in range(reps))
    lo, hi = ds[int(0.025 * (reps - 1))], ds[int(math.ceil(0.975 * (reps - 1)))]
    return {"delta": point, "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0), "n_base": sum(len(g) for g in b), "n_post": sum(len(g) for g in p),
            "sessions_base": len(b), "sessions_post": len(p)}


# ----------------------------------------------------------------------------------------------------------------- one transcript file
def _toks(u: dict) -> list[int]:
    return [u.get("input_tokens") or 0, u.get("output_tokens") or 0, u.get("cache_creation_input_tokens") or 0, u.get("cache_read_input_tokens") or 0]


def parse_transcript(records, is_sub: bool) -> dict:
    """Fold the records of one transcript file into counts and timings. ``records`` is an iterable of already-decoded dicts.
    Only numbers, timestamps and tool names are kept."""
    turns: list[dict] = []
    pre = {"tok": [0, 0, 0, 0], "tools": 0, "rgg": 0, "bs": 0}
    usage: dict[str, list[int]] = {}
    usage_turn: dict[str, int] = {}
    has_content: dict[str, bool] = {}
    tool_counts: Counter = Counter()
    seen_tools: dict[str, tuple] = {}            # tool_use id -> (name, ts, turn idx)
    gaps: dict[str, list] = defaultdict(list)
    dispatches: dict[str, dict] = {}
    compactions: list[str] = []
    hook_groups: dict[tuple, list] = {}          # (toolUseID, event) -> [max_ms, turn idx, timed_out_or_error]
    hook_events: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0])   # event -> [sum_ms, n, errors, timeouts]
    stop_ms: list[tuple[int, int]] = []
    first_ts = last_ts = None
    cwd = None
    bash_search = 0

    for o in records:
        if not isinstance(o, dict):
            continue
        ts = o.get("timestamp")
        if ts:
            first_ts = first_ts or ts
            if not last_ts or ts > last_ts:
                last_ts = ts
        if o.get("cwd"):
            cwd = o["cwd"]
        typ = o.get("type")
        cur = len(turns) - 1
        if typ == "system":
            st = o.get("subtype")
            if st == "compact_boundary":
                compactions.append(ts)
            elif st == "stop_hook_summary":
                infos = o.get("hookInfos")
                if isinstance(infos, list):
                    ms = [int(i.get("durationMs") or 0) for i in infos if isinstance(i, dict)]
                    if ms:
                        stop_ms.append((cur, max(ms)))
            continue
        if typ == "attachment":
            a = o.get("attachment") or {}
            if a.get("type") in HOOK_TYPES:
                ev = a.get("hookEvent") or "?"
                try:
                    ms = int(float(a.get("durationMs") or 0))
                except (TypeError, ValueError):
                    ms = 0
                bad = a["type"] != "hook_success"
                to = bool(a.get("timedOut"))
                h = hook_events[ev]
                h[0] += ms
                h[1] += 1
                h[2] += int(bad)
                h[3] += int(to)
                if ev != "SessionStart" and ev != "SessionEnd":
                    g = hook_groups.setdefault((a.get("toolUseID") or o.get("uuid"), ev), [0, cur, False])
                    g[0] = max(g[0], ms)
                    g[2] = g[2] or bad
            continue
        msg = o.get("message") or {}
        if typ == "user":
            if not is_sub:
                real, _ = _is_real_user(o)
                if real:
                    turns.append({"ts": ts, "end": ts, "tools": 0, "rgg": 0, "bs": 0, "tok": [0, 0, 0, 0], "hook_wall_ms": 0, "hook_n": 0, "stop_ms": 0})
                    continue
            c = msg.get("content")
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        tu = seen_tools.get(b.get("tool_use_id"))
                        if tu and ts and tu[1]:
                            gaps[tu[0]].append(int(round(max(0.0, parse_ts(ts) - parse_ts(tu[1])) * 1000)))
                        if cur >= 0 and ts and ts > turns[cur]["end"]:
                            turns[cur]["end"] = ts
            continue
        if typ == "assistant":
            if msg.get("model") == "<synthetic>":
                continue
            mid = msg.get("id") or o.get("uuid")
            u = msg.get("usage")
            content = msg.get("content") or []
            blocks = [b for b in content if isinstance(b, dict)]
            if u and mid:
                t = _toks(u)
                prev = usage.get(mid)
                usage[mid] = [t[0], max(prev[1] if prev else 0, t[1]), t[2], t[3]]
                usage_turn.setdefault(mid, cur)
                has_content[mid] = has_content.get(mid, False) or any(b.get("type") in ("text", "tool_use") for b in blocks)
            if cur >= 0 and ts and ts > turns[cur]["end"]:
                turns[cur]["end"] = ts
            for b in blocks:
                if b.get("type") != "tool_use":
                    continue
                tid, name = b.get("id"), b.get("name") or "?"
                if tid in seen_tools:
                    continue
                seen_tools[tid] = (name, ts, cur)
                tool_counts[name] += 1
                tgt = turns[cur] if cur >= 0 else pre
                tgt["tools"] += 1
                if name in SEARCH_TOOLS:
                    tgt["rgg"] += 1
                elif name == "Bash" and _BASH_SEARCH.search(str((b.get("input") or {}).get("command") or "")):
                    tgt["bs"] += 1
                    bash_search += 1
                if name in DISPATCH_TOOLS:
                    dispatches[tid] = {"ts": ts, "type": (b.get("input") or {}).get("subagent_type") or "general-purpose", "turn": cur}

    tot = [0, 0, 0, 0]
    for mid, t in usage.items():
        tgt = turns[usage_turn[mid]]["tok"] if usage_turn[mid] >= 0 else pre["tok"]
        for i in range(4):
            tgt[i] += t[i]
            tot[i] += t[i]
    for (_, _ev), (ms, ti, _bad) in hook_groups.items():
        if ti >= 0:
            turns[ti]["hook_wall_ms"] += ms
            turns[ti]["hook_n"] += 1
    hook_wall_all = sum(g[0] for g in hook_groups.values())
    hook_n_all = len(hook_groups)
    for ti, ms in stop_ms:
        if ti >= 0:
            turns[ti]["stop_ms"] += ms
    tiny = sum(1 for m, t in usage.items() if t[1] <= 3 and has_content.get(m))
    return {
        "first_ts": first_ts, "last_ts": last_ts, "cwd_name": re.split(r"[\\/]", cwd.rstrip("\\/"))[-1] if cwd else None,
        "turns": turns, "pre": pre, "tokens": tot, "api_calls": len(usage), "out_tiny": tiny,
        "tool_counts": dict(tool_counts), "n_tools": sum(tool_counts.values()), "rgg": sum(tool_counts[t] for t in SEARCH_TOOLS), "bash_search": bash_search,
        "dispatches": dispatches, "compactions": compactions, "hook_wall_ms": hook_wall_all, "hook_n": hook_n_all,
        "hook_events": {k: v for k, v in hook_events.items()}, "stop_ms_sum": sum(m for _, m in stop_ms),
        "gaps": {k: v[:GAP_CAP] for k, v in gaps.items() if k in GAP_TOOLS},
    }


def _read_records(path: Path):
    yield from iter_jsonl(path)


def load_file_summary(path: Path, rel: str, cache_dir: Path) -> dict:
    """Cached per-file summary (re-parsed when the file's size or mtime changed)."""
    cp = cache_dir / f"{sha16(rel)}.json"
    st = path.stat()
    c = read_json(cp, None)
    if c and c.get("v") == CACHE_VERSION and c.get("size") == st.st_size and c.get("mtime_ns") == st.st_mtime_ns:
        return c
    parts = rel.split("/")
    is_sub = len(parts) == 4
    s = parse_transcript(_read_records(path), is_sub)
    meta = {}
    agent_id = None
    if is_sub:
        m = re.match(r"agent-([A-Za-z0-9]+)\.jsonl$", parts[3])
        agent_id = m.group(1) if m else parts[3]
        meta = read_json(path.with_name(path.name.removesuffix(".jsonl") + ".meta.json"), {}) or {}
    s.update({"v": CACHE_VERSION, "rel": rel, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "project_dir": parts[0],
              "session": parts[1].removesuffix(".jsonl"), "agent_id": agent_id, "is_sub": is_sub,
              "tool_use_id": meta.get("toolUseId"), "agent_type": meta.get("agentType")})
    atomic_write_json(cp, s)
    return s


# ----------------------------------------------------------------------------------------------------------------- units and rows
def _day(ts: str | None) -> str:
    return (ts or "")[:10]


def _wall(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    return max(0.0, parse_ts(b) - parse_ts(a))


def fresh(t) -> int:
    """input + output + cache creation (everything that is not a cache read)."""
    return t[0] + t[1] + t[2]


def load_labels(outcomes_path: Path) -> dict[str, str]:
    """dispatch tool_use id -> outcome label from the ingest labeler (extract.classify); continuations are skipped."""
    out = {}
    for r in iter_jsonl(outcomes_path):
        if r.get("kind") == "dispatch" and str(r.get("id", "")).startswith("out:"):
            out[r["id"][4:]] = r.get("outcome")
    return out


def build_units(summaries: list[dict], labels: dict[str, str], cutoff: float):
    """Flatten file summaries into per-session turn units and dispatch units. ``post`` = started at/after the cutoff."""
    sessions: dict[tuple, dict] = {}
    for s in summaries:
        key = (s["project_dir"], s["session"])
        ss = sessions.setdefault(key, {"main": None, "subs": []})
        if s["is_sub"]:
            ss["subs"].append(s)
        else:
            ss["main"] = s
    turn_units, disp_units, rows = [], [], []
    for (pdir, sid), ss in sorted(sessions.items()):
        main, subs = ss["main"], ss["subs"]
        files = ([main] if main else []) + subs
        by_tid = {x["tool_use_id"]: x for x in subs if x.get("tool_use_id")}
        starts = [f["first_ts"] for f in files if f["first_ts"]]
        ends = [f["last_ts"] for f in files if f["last_ts"]]
        start, end = (min(starts) if starts else None), (max(ends) if ends else None)
        proj = (main or (subs[0] if subs else {})).get("cwd_name") or pdir
        sess_turns, sess_disp = [], []
        for i, t in enumerate((main or {}).get("turns") or []):
            w = _wall(t["ts"], t["end"])
            u = {"session": sid, "day": _day(t["ts"]), "post": parse_ts(t["ts"]) >= cutoff, "total": sum(t["tok"]), "fresh": fresh(t["tok"]),
                 "out": t["tok"][1], "wall": w, "tools": t["tools"], "hook_wall_ms": t["hook_wall_ms"], "stop_ms": t["stop_ms"]}
            sess_turns.append(u)
        for f in files:
            for tid, d in f["dispatches"].items():
                sf = by_tid.get(tid)
                lab = labels.get(tid)
                u = {"session": sid, "day": _day(d["ts"]), "post": parse_ts(d["ts"]) >= cutoff, "type": d["type"], "label": lab,
                     "total": sum(sf["tokens"]) if sf else None, "fresh": fresh(sf["tokens"]) if sf else None,
                     "wall": _wall(sf["first_ts"], sf["last_ts"]) if sf else None, "tools": sf["n_tools"] if sf else None,
                     "rgg": sf["rgg"] if sf else None, "bs": sf["bash_search"] if sf else None,
                     "search": (sf["rgg"] + sf["bash_search"]) if sf else None, "hook_wall_ms": sf["hook_wall_ms"] if sf else None}
                sess_disp.append(u)
        gaps_fast = [g for f in files for k, v in f["gaps"].items() if k in FAST_TOOLS for g in v]
        tok = [sum(f["tokens"][i] for f in files) for i in range(4)]
        ltok = (main or {}).get("tokens") or [0, 0, 0, 0]
        stok = [sum(f["tokens"][i] for f in subs) for i in range(4)]
        prompt = tok[0] + tok[2] + tok[3]
        labd = [u["label"] for u in sess_disp if u["label"]]
        rows.append({
            "session": sid, "project": proj, "day": _day(start), "start": start, "end": end,
            "baseline": parse_ts(start) < cutoff, "n_turns": len(sess_turns), "n_dispatch": len(sess_disp), "n_subagent_files": len(subs),
            "tok_leader": {"input": ltok[0], "output": ltok[1], "cache_write": ltok[2], "cache_read": ltok[3]},
            "tok_subagents": {"input": stok[0], "output": stok[1], "cache_write": stok[2], "cache_read": stok[3]},
            "tok_total": sum(tok), "cache_read_share": (tok[3] / prompt) if prompt else None,
            "out_tiny_share": (sum(f["out_tiny"] for f in files) / max(1, sum(f["api_calls"] for f in files))),
            "turn_tokens_med": median([u["total"] for u in sess_turns]) if sess_turns else None,
            "turn_fresh_med": median([u["fresh"] for u in sess_turns]) if sess_turns else None,
            "turn_wall_s_med": median([u["wall"] for u in sess_turns if u["wall"] is not None]) if sess_turns else None,
            "turn_wall_s_p90": p90([u["wall"] for u in sess_turns if u["wall"] is not None]) if sess_turns else None,
            "tool_calls_per_turn_med": median([u["tools"] for u in sess_turns]) if sess_turns else None,
            "disp_tokens_med": median([u["total"] for u in sess_disp if u["total"] is not None]) if sess_disp else None,
            "disp_fresh_med": median([u["fresh"] for u in sess_disp if u["fresh"] is not None]) if sess_disp else None,
            "disp_wall_s_med": median([u["wall"] for u in sess_disp if u["wall"] is not None]) if sess_disp else None,
            "disp_wall_s_p90": p90([u["wall"] for u in sess_disp if u["wall"] is not None]) if sess_disp else None,
            "disp_rgg_med": median([u["rgg"] for u in sess_disp if u["rgg"] is not None]) if sess_disp else None,
            "disp_tools_med": median([u["tools"] for u in sess_disp if u["tools"] is not None]) if sess_disp else None,
            "hook_wall_ms_per_turn_med": median([u["hook_wall_ms"] for u in sess_turns]) if sess_turns else None,
            "stop_hook_ms_per_turn_med": median([u["stop_ms"] for u in sess_turns]) if sess_turns else None,
            "read_gap_ms_med": median(gaps_fast) if gaps_fast else None,
            "n_labeled": len(labd), "n_redispatch": labd.count("re-dispatch"), "n_fixround": labd.count("fix-round"),
            "compactions": sum(len(f["compactions"]) for f in files),
        })
        turn_units += sess_turns
        disp_units += sess_disp
    return rows, turn_units, disp_units


# ----------------------------------------------------------------------------------------------------------------- aggregation
def _vals(units, key, post=None):
    return [u[key] for u in units if u.get(key) is not None and (post is None or u["post"] == post)]


def _by_session(units, key, post):
    d = defaultdict(list)
    for u in units:
        if u["post"] == post and u.get(key) is not None:
            d[u["session"]].append(u[key])
    return d


def headline(rows, turn_units, disp_units, gap_samples, post=False) -> dict:
    """Medians and p90s over units on one side of the cutoff."""
    T = [u for u in turn_units if u["post"] == post]
    D = [u for u in disp_units if u["post"] == post]
    R = [r for r in rows if r["baseline"] != post]
    tok = defaultdict(int)
    for r in R:
        for part in ("tok_leader", "tok_subagents"):
            for k, v in r[part].items():
                tok[k] += v
    prompt = tok["input"] + tok["cache_write"] + tok["cache_read"]
    lab = [u["label"] for u in D if u["label"]]
    lab_tok = defaultdict(int)
    for u in D:
        if u["label"] and u["total"]:
            lab_tok[u["label"]] += u["total"]
    tot_lab_tok = sum(lab_tok.values())
    h = {
        "sessions": len(R), "turns": len(T), "dispatches": len(D), "dispatches_with_tokens": len(_vals(D, "total")),
        "tokens_per_turn_total": {"med": median(_vals(T, "total")), "p90": p90(_vals(T, "total"))},
        "tokens_per_turn_fresh": {"med": median(_vals(T, "fresh")), "p90": p90(_vals(T, "fresh"))},
        "tokens_per_dispatch_total": {"med": median(_vals(D, "total")), "p90": p90(_vals(D, "total"))},
        "tokens_per_dispatch_fresh": {"med": median(_vals(D, "fresh")), "p90": p90(_vals(D, "fresh"))},
        "wall_s_per_turn": {"med": median(_vals(T, "wall")), "p90": p90(_vals(T, "wall"))},
        "wall_s_per_dispatch": {"med": median(_vals(D, "wall")), "p90": p90(_vals(D, "wall"))},
        "tool_calls_per_turn": {"med": median(_vals(T, "tools")), "p90": p90(_vals(T, "tools"))},
        "tool_calls_per_dispatch": {"med": median(_vals(D, "tools")), "p90": p90(_vals(D, "tools"))},
        "read_grep_glob_per_dispatch": {"med": median(_vals(D, "rgg")), "p90": p90(_vals(D, "rgg"))},
        "bash_search_per_dispatch": {"med": median(_vals(D, "bs")), "p90": p90(_vals(D, "bs"))},
        "search_total_per_dispatch": {"med": median(_vals(D, "search")), "p90": p90(_vals(D, "search"))},
        "hook_wall_ms_per_turn": {"med": median(_vals(T, "hook_wall_ms")), "p90": p90(_vals(T, "hook_wall_ms"))},
        "stop_hook_ms_per_turn": {"med": median(_vals(T, "stop_ms")), "p90": p90(_vals(T, "stop_ms"))},
        "labeled_dispatches": len(lab),
        "redispatch_rate": (lab.count("re-dispatch") / len(lab)) if lab else None,
        "fixround_rate": (lab.count("fix-round") / len(lab)) if lab else None,
        "redispatch_token_share": (lab_tok["re-dispatch"] / tot_lab_tok) if tot_lab_tok else None,
        "fixround_token_share": (lab_tok["fix-round"] / tot_lab_tok) if tot_lab_tok else None,
        "compactions": sum(r["compactions"] for r in R),
        "compactions_per_session": (sum(r["compactions"] for r in R) / len(R)) if R else None,
        "cache_read_share": (tok["cache_read"] / prompt) if prompt else None,
        "out_tiny_share": median([r["out_tiny_share"] for r in R]) if R else None,
    }
    g = [x for x in gap_samples if x["post"] == post]
    h["tool_gap_ms"] = {}
    for tool in GAP_TOOLS:
        gv = [v for x in g for v in x["g"].get(tool, [])]
        h["tool_gap_ms"][tool] = {"med": median(gv), "p90": p90(gv), "n": len(gv)}
    return h


METRICS = [  # (label, unit set, key, scale, unit)
    ("tokens per turn (all, incl. cache read)", "T", "total", 1, "tok"),
    ("tokens per turn (fresh: in+out+cache write)", "T", "fresh", 1, "tok"),
    ("tokens per dispatch (all)", "D", "total", 1, "tok"),
    ("tokens per dispatch (fresh)", "D", "fresh", 1, "tok"),
    ("wall per turn", "T", "wall", 1, "s"),
    ("wall per dispatch", "D", "wall", 1, "s"),
    ("tool calls per turn", "T", "tools", 1, ""),
    ("tool calls per dispatch", "D", "tools", 1, ""),
    ("Read+Grep+Glob per dispatch", "D", "rgg", 1, ""),
    ("search-like Bash (grep/rg/find/cat/ls...) per dispatch", "D", "bs", 1, ""),
    ("all searching per dispatch", "D", "search", 1, ""),
    ("hook wall per turn", "T", "hook_wall_ms", 1, "ms"),
]


def deltas(turn_units, disp_units, reps=1000) -> list[dict]:
    out = []
    for label, which, key, _sc, unit in METRICS:
        U = turn_units if which == "T" else disp_units
        b, p = _by_session(U, key, False), _by_session(U, key, True)
        for stat_name, stat in (("median", median), ("p90", p90)):
            r = cluster_bootstrap_delta(b, p, stat=stat, reps=reps)
            out.append({"metric": label, "stat": stat_name, "unit": unit, **(r or {"delta": None})})
    return out


def daily(rows, turn_units, disp_units) -> list[dict]:
    days = sorted({r["day"] for r in rows if r["day"]} | {u["day"] for u in list(turn_units) + list(disp_units) if u["day"]})
    res = []
    for d in days:
        R = [r for r in rows if r["day"] == d]
        T = [u for u in turn_units if u["day"] == d]
        D = [u for u in disp_units if u["day"] == d]
        lab = [u["label"] for u in D if u["label"]]
        tok = defaultdict(int)
        for r in R:
            for part in ("tok_leader", "tok_subagents"):
                for k, v in r[part].items():
                    tok[k] += v
        prompt = tok["input"] + tok["cache_write"] + tok["cache_read"]
        res.append({
            "day": d, "sessions": len(R), "turns": len(T), "dispatches": len(D),
            "turn_tok_med": median(_vals(T, "total")), "turn_fresh_med": median(_vals(T, "fresh")),
            "disp_tok_med": median(_vals(D, "total")), "disp_fresh_med": median(_vals(D, "fresh")),
            "turn_wall_med": median(_vals(T, "wall")), "disp_wall_med": median(_vals(D, "wall")),
            "tools_turn_med": median(_vals(T, "tools")), "rgg_disp_med": median(_vals(D, "rgg")),
            "hook_ms_turn_med": median(_vals(T, "hook_wall_ms")),
            "redispatch_pct": (100 * lab.count("re-dispatch") / len(lab)) if lab else None,
            "fixround_pct": (100 * lab.count("fix-round") / len(lab)) if lab else None,
            "labeled": len(lab), "compactions": sum(r["compactions"] for r in R),
            "cache_read_pct": (100 * tok["cache_read"] / prompt) if prompt else None,
            "post": bool(R) and all(not r["baseline"] for r in R) if R else bool(T or D) and all(u["post"] for u in T + D),
        })
    return res


# ----------------------------------------------------------------------------------------------------------------- rendering
def _f(x, digits=0, pct=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if pct:
        return f"{x:.{max(digits, 1)}f}%"
    a = abs(x)
    if digits == 0 and a >= 10000:
        return f"{x / 1000:.0f}k" if a < 1e6 else f"{x / 1e6:.1f}M"
    return f"{x:,.{digits}f}"


def render_md(cut_iso: str, now_iso: str, base: dict, post: dict, dl: list[dict], days: list[dict]) -> str:
    L = [f"# Cross-session scorecard", "",
         f"Generated {now_iso} by `python -m ingest scorecard`. Baseline cutoff: **{cut_iso}** ({BASELINE_NOTE}). "
         "Numbers are aggregates over local Claude Code transcripts (leader sessions plus subagent transcripts); no text is included.", "",
         "## Baseline headline (before the cutoff)", "",
         f"{base['sessions']} sessions, {base['turns']} user turns, {base['dispatches']} dispatches "
         f"({base['dispatches_with_tokens']} with a subagent transcript).", "",
         "| metric | median | p90 |", "| --- | ---: | ---: |"]

    def row(name, k, digits=0):
        v = base[k]
        L.append(f"| {name} | {_f(v['med'], digits)} | {_f(v['p90'], digits)} |")
    row("tokens per user turn, all (leader)", "tokens_per_turn_total")
    row("tokens per user turn, fresh = in+out+cache write (leader)", "tokens_per_turn_fresh")
    row("tokens per subagent dispatch, all", "tokens_per_dispatch_total")
    row("tokens per subagent dispatch, fresh", "tokens_per_dispatch_fresh")
    row("wall seconds per user turn", "wall_s_per_turn", 1)
    row("wall seconds per subagent dispatch", "wall_s_per_dispatch", 1)
    row("tool calls per user turn", "tool_calls_per_turn")
    row("tool calls per dispatch", "tool_calls_per_dispatch")
    row("Read+Grep+Glob calls per dispatch (searching)", "read_grep_glob_per_dispatch")
    row("search-like Bash calls (grep/rg/find/cat/ls/head/tail) per dispatch", "bash_search_per_dispatch")
    row("all searching calls per dispatch (Read+Grep+Glob+search-like Bash)", "search_total_per_dispatch")
    row("hook wall ms per turn (Pre/PostToolUse, prompt hooks; max per event group)", "hook_wall_ms_per_turn")
    row("Stop-hook ms per turn (max hook per stop)", "stop_hook_ms_per_turn")
    for tool, fg in base["tool_gap_ms"].items():
        L.append(f"| tool_use to tool_result gap, {tool}, ms (n={fg['n']}) | {_f(fg['med'])} | {_f(fg['p90'])} |")
    L += ["",
          f"- re-dispatch rate {_f(base['redispatch_rate'] and 100 * base['redispatch_rate'], 1, True)} of {base['labeled_dispatches']} labeled dispatches "
          f"({_f(base['redispatch_token_share'] and 100 * base['redispatch_token_share'], 1, True)} of labeled-dispatch tokens); "
          f"fix-round rate {_f(base['fixround_rate'] and 100 * base['fixround_rate'], 1, True)} "
          f"({_f(base['fixround_token_share'] and 100 * base['fixround_token_share'], 1, True)} of tokens). Labels are the ingest labeler's heuristic (a proxy).",
          f"- compaction events: {base['compactions']} ({_f(base['compactions_per_session'], 2)} per session).",
          f"- cache reads are {_f(base['cache_read_share'] and 100 * base['cache_read_share'], 1, True)} of prompt tokens (input + cache write + cache read).",
          f"- output-token under-recording check: {_f(base['out_tiny_share'] and 100 * base['out_tiny_share'], 1, True)} of API messages that carry text or a tool call "
          "have output_tokens <= 3 (a streamed placeholder); sampled recorded output tokens exceed a content-length estimate, so no under-recording is detected by this check. "
          "Known gaps instead: rejected/aborted requests leave no usage record, thinking tokens are in usage but not in the visible content, and a leader turn excludes its subagents' tokens (counted per dispatch).",
          "- Hook overhead is read two ways: the hook attachment durations above (parallel hooks of one event count once, as their max), and the tool gaps: "
          "Read returns in tens of ms (the no-hook floor), so the excess on Edit/Write/Bash is hooks plus permission waits plus real work, not Read-style latency.",
          "- turn wall time is active time (user message to last assistant/tool-result record) and includes waiting on permission prompts; "
          "the leader's tokens are per turn, subagent tokens are per dispatch (see the session rows for totals).", ""]
    if post["turns"] or post["dispatches"]:
        L += [f"## After the cutoff", "", f"{post['sessions']} sessions, {post['turns']} turns, {post['dispatches']} dispatches.", "",
              "| metric | stat | after - baseline | 95% CI (session bootstrap) | n base / n after | sessions base / after |", "| --- | --- | ---: | --- | --- | --- |"]
        for d in dl:
            if d.get("delta") is None:
                L.append(f"| {d['metric']} | {d['stat']} | not enough sessions | | | |")
            else:
                mark = " **" if d["sig"] else ""
                L.append(f"| {d['metric']} | {d['stat']} | {d['delta']:+,.1f} {d['unit']}{mark} | [{d['lo']:+,.1f}, {d['hi']:+,.1f}] | "
                         f"{d['n_base']} / {d['n_post']} | {d['sessions_base']} / {d['sessions_post']} |")
        L += ["", "A row in bold has a CI that excludes 0. Units inside a session are correlated, so sessions are resampled, not turns.", ""]
    else:
        L += ["## After the cutoff", "", "No sessions after the cutoff yet. Re-run later: this section fills with after-minus-baseline deltas and bootstrap CIs.", ""]
    L += ["## Trend by day (UTC)", "",
          "| day | sess | turns | disp | tok/turn med (fresh) | tok/disp med (fresh) | s/turn | s/disp | tools/turn | R+G+G/disp | hook ms/turn | redisp % | fix % | compact | cache % | |",
          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for d in days:
        L.append(f"| {d['day']} | {d['sessions']} | {d['turns']} | {d['dispatches']} | {_f(d['turn_tok_med'])} ({_f(d['turn_fresh_med'])}) | "
                 f"{_f(d['disp_tok_med'])} ({_f(d['disp_fresh_med'])}) | {_f(d['turn_wall_med'], 1)} | {_f(d['disp_wall_med'], 0)} | {_f(d['tools_turn_med'])} | "
                 f"{_f(d['rgg_disp_med'])} | {_f(d['hook_ms_turn_med'])} | {_f(d['redispatch_pct'], 0, True)} | {_f(d['fixround_pct'], 0, True)} | "
                 f"{d['compactions']} | {_f(d['cache_read_pct'], 0, True)} | {'after' if d['post'] else ''} |")
    L += ["", "Per-session rows: `data/lifelong/scorecard.jsonl` (git-ignored, derived from private transcripts).", ""]
    return "\n".join(L)


def render_html(cut_iso: str, now_iso: str, base: dict, days: list[dict]) -> str:
    """Self-contained trend page: inline CSS + inline SVG sparklines, no scripts, no external requests."""
    series = [("Tokens per user turn, median (fresh)", "turn_fresh_med", ""), ("Tokens per dispatch, median (fresh)", "disp_fresh_med", ""),
              ("Wall seconds per user turn, median", "turn_wall_med", "s"), ("Wall seconds per dispatch, median", "disp_wall_med", "s"),
              ("Read+Grep+Glob calls per dispatch, median", "rgg_disp_med", ""), ("Hook ms per turn, median", "hook_ms_turn_med", "ms"),
              ("Re-dispatch rate", "redispatch_pct", "%"), ("Fix-round rate", "fixround_pct", "%"), ("Cache-read share", "cache_read_pct", "%")]
    W, H, P = 560, 120, 8

    def spark(key):
        pts = [(i, d[key], d["post"]) for i, d in enumerate(days) if d[key] is not None and not (isinstance(d[key], float) and math.isnan(d[key]))]
        if len(pts) < 1:
            return "<p class=m>no data</p>"
        ys = [p[1] for p in pts]
        lo, hi = min(ys + [0]), max(ys)
        span = (hi - lo) or 1
        n = max(1, len(days) - 1)
        xy = [(P + (W - 2 * P) * i / n, H - P - (H - 2 * P) * (v - lo) / span, post) for i, v, post in pts]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in xy)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="{"a" if post else "b"}"/>' for x, y, post in xy)
        return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{key} by day"><polyline points="{line}" fill="none" class="l"/>{dots}'
                f'<text x="{P}" y="12" class="t">max {_f(hi, 1)}</text></svg>')
    cards = "".join(f"<section><h2>{t}</h2>{spark(k)}</section>" for t, k, _ in series)
    first, last = (days[0]["day"], days[-1]["day"]) if days else ("-", "-")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Session scorecard</title><style>
:root{{--bg:#fff;--fg:#1a1a1a;--m:#666;--l:#2b6cb0;--b:#2b6cb0;--a:#c05621;--c:#f6f6f6}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16181c;--fg:#e8e8e8;--m:#9aa;--l:#63b3ed;--b:#63b3ed;--a:#f6ad55;--c:#20242a}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:640px;padding:16px}}
h1{{font-size:20px;margin:0 0 4px}}h2{{font-size:14px;margin:0 0 4px}}.m{{color:var(--m);font-size:13px}}
section{{background:var(--c);border-radius:8px;padding:10px 12px;margin:12px 0}}svg{{width:100%;height:auto}}
.l{{stroke:var(--l);stroke-width:2}}.b{{fill:var(--b)}}.a{{fill:var(--a)}}.t{{fill:var(--m);font-size:11px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border-bottom:1px solid var(--c);padding:3px 6px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
</style></head><body><h1>Session scorecard</h1>
<p class=m>Generated {now_iso}. Days {first} to {last} (UTC). Blue dots: before the baseline cutoff ({cut_iso}); orange dots: after it.
Aggregates only. Daily medians over turns and dispatches; the dated table with confidence intervals is in <code>results/scorecard.md</code>.</p>
<section><h2>Baseline headline</h2><table><tr><th>metric</th><th>median</th><th>p90</th></tr>
<tr><td>tokens / user turn (fresh)</td><td>{_f(base['tokens_per_turn_fresh']['med'])}</td><td>{_f(base['tokens_per_turn_fresh']['p90'])}</td></tr>
<tr><td>tokens / dispatch (fresh)</td><td>{_f(base['tokens_per_dispatch_fresh']['med'])}</td><td>{_f(base['tokens_per_dispatch_fresh']['p90'])}</td></tr>
<tr><td>wall s / user turn</td><td>{_f(base['wall_s_per_turn']['med'], 1)}</td><td>{_f(base['wall_s_per_turn']['p90'], 1)}</td></tr>
<tr><td>wall s / dispatch</td><td>{_f(base['wall_s_per_dispatch']['med'], 0)}</td><td>{_f(base['wall_s_per_dispatch']['p90'], 0)}</td></tr>
<tr><td>Read+Grep+Glob / dispatch</td><td>{_f(base['read_grep_glob_per_dispatch']['med'])}</td><td>{_f(base['read_grep_glob_per_dispatch']['p90'])}</td></tr>
</table></section>{cards}</body></html>
"""


# ----------------------------------------------------------------------------------------------------------------- driver
def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def run_scorecard(cfg: Config, results_dir: Path, reset_baseline: bool = False, reps: int = 1000, quiet: bool = False, now: float | None = None) -> dict:
    now_ts = now if now is not None else datetime.now(timezone.utc).timestamp()
    bpath = cfg.data / "scorecard_baseline.json"
    b = read_json(bpath, None)
    if reset_baseline or not b or "cutoff_ts" not in b:
        b = {"cutoff_ts": _iso(now_ts), "note": BASELINE_NOTE}
        atomic_write_json(bpath, b)
    cutoff = parse_ts(b["cutoff_ts"])
    cache_dir = cfg.data / "scorecard_state"
    files = discover(cfg.projects)
    summaries = []
    for p, rel in files:
        try:
            summaries.append(load_file_summary(p, rel, cache_dir))
        except OSError:
            continue
    labels = load_labels(cfg.outcomes)
    rows, T, D = build_units(summaries, labels, cutoff)
    gap_samples = [{"post": parse_ts(x["first_ts"]) >= cutoff, "g": x["gaps"]} for x in summaries if x["gaps"]]
    base, post = headline(rows, T, D, gap_samples, False), headline(rows, T, D, gap_samples, True)
    dl = deltas(T, D, reps=reps) if (post["turns"] or post["dispatches"]) else []
    days = daily(rows, T, D)
    results_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.data / "scorecard.jsonl", "\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows) + "\n")
    now_iso = _iso(now_ts)
    atomic_write_text(results_dir / "scorecard.md", render_md(b["cutoff_ts"], now_iso, base, post, dl, days))
    atomic_write_text(results_dir / "scorecard.html", render_html(b["cutoff_ts"], now_iso, base, days))
    atomic_write_json(cfg.data / "scorecard_headline.json", {"cutoff": b["cutoff_ts"], "generated": now_iso, "baseline": base, "after": post})
    res = {"files": len(files), "sessions": len(rows), "turns": len(T), "dispatches": len(D), "cutoff": b["cutoff_ts"],
           "after_turns": post["turns"], "after_dispatches": post["dispatches"]}
    return res
