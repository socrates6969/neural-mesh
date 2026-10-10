"""Incremental, resumable reader of Claude Code transcripts.

Every transcript file (session main file or subagent sidechain) has its own byte cursor and its own compact, SCRUBBED
per-file state under ``state/files/<sha16(rel)>.json``. A run reads only the complete new lines (up to the last newline)
since the cursor, folds them into the file state, and saves state + cursor atomically. A file that shrank or whose head
changed is re-read from byte 0 with a fresh state (outputs are de-duplicated by id, so this is safe).
Nothing raw is stored: tool inputs and tool results are never kept, except the scrubbed head of Agent results.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from .scrub import scrub, touches_sensitive_path
from .util import atomic_write_json, read_json, sha16

STATE_VERSION = 1
MAX_PROMPT = 6000
MAX_RESP = 10000
MAX_RESULT_HEAD = 600
DISPATCH_TOOLS = {"Agent", "Task"}
_PATH_KEYS = ("file_path", "path", "notebook_path", "pattern", "glob", "command", "url", "cwd", "directory", "dir", "paths", "files")
_SYSTEM_PREFIX = re.compile(r"^\s*(?:<(?:task-notification|command-name|command-message|command-args|local-command|system-reminder|bash-|user-prompt-submit-hook)|Caveat:|\[Request interrupted)")
_SYS_REMINDER = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")
_API_ERR = re.compile(r"(?i)API Error|rate limit|weekly limit|usage limit|overloaded|\b429\b")
_TURN_LIMIT = re.compile(r"(?i)maximum number of turns|max[_ ]turns|turn limit")


def discover(projects: Path) -> list[tuple[Path, str]]:
    """All transcript files with their path relative to ``projects`` (sorted, deterministic)."""
    out = []
    if not projects.exists():
        return out
    for p in sorted(projects.rglob("*.jsonl")):
        rel = p.relative_to(projects).as_posix()
        parts = rel.split("/")
        if len(parts) == 2 or (len(parts) == 4 and parts[2] == "subagents"):
            out.append((p, rel))
    return out


def _new_state(rel: str) -> dict:
    parts = rel.split("/")
    agent_id = None
    if len(parts) == 4:
        m = re.match(r"agent-([A-Za-z0-9]+)\.jsonl$", parts[3])
        agent_id = m.group(1) if m else parts[3]
    return {
        "v": STATE_VERSION, "rel": rel, "project_dir": parts[0], "session": parts[1].removesuffix(".jsonl"),
        "agent_id": agent_id, "meta": None,
        "cursor": {"offset": 0, "size": 0, "mtime_ns": 0, "head": ""},
        "cwd": None, "first_ts": None, "last_ts": None,
        "dispatches": {}, "results": {}, "usage": {}, "models": [],
        "turns": [], "handback": None, "signals": {}, "tainted_tools": 0, "n_tool_calls": 0,
        "scrub_hits": {},
    }


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _is_real_user(o: dict) -> tuple[bool, str]:
    if o.get("type") != "user" or o.get("isMeta") or o.get("isCompactSummary"):
        return False, ""
    c = (o.get("message") or {}).get("content")
    if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
        return False, ""
    t = _SYS_REMINDER.sub("", _text_of(c)).strip()
    if not t or _SYSTEM_PREFIX.match(t):
        return False, ""
    return True, t


def _path_fields(inp) -> str:
    if not isinstance(inp, dict):
        return ""
    vals = []
    for k in _PATH_KEYS:
        v = inp.get(k)
        if isinstance(v, str):
            vals.append(v)
        elif isinstance(v, list):
            vals.extend(x for x in v if isinstance(x, str))
    return "\n".join(vals)


def _hits(st: dict, c: Counter) -> None:
    h = st["scrub_hits"]
    for k, v in c.items():
        h[k] = h.get(k, 0) + v


def _scrub(st: dict, text: str, limit: int) -> dict:
    """Scrub and cap a text; keep the secret flag so sft/memory can drop it."""
    r = scrub(text[: limit * 2] if text else "")
    _hits(st, r.hits)
    t = r.text
    if len(t) > limit:
        t = t[: limit // 3] + "\n[...]\n" + t[-(limit - limit // 3):]
    return {"t": t, "s": r.secret}


def _cur_turn(st: dict) -> dict | None:
    return st["turns"][-1] if st["turns"] else None


def apply_record(st: dict, o: dict) -> None:
    ts = o.get("timestamp")
    if ts:
        st["first_ts"] = st["first_ts"] or ts
        if not st["last_ts"] or ts > st["last_ts"]:
            st["last_ts"] = ts
    if o.get("cwd"):
        st["cwd"] = o["cwd"]
    typ = o.get("type")
    msg = o.get("message") or {}

    real, utext = _is_real_user(o)
    if real:
        prev = _cur_turn(st)
        if prev is not None and prev.get("fb") is None:
            prev["fb"] = scrub(utext[:200]).text  # first words of the NEXT user message: a feedback signal for the previous turn
        st["turns"].append({"ts": ts, "user": _scrub(st, utext, MAX_PROMPT), "resp": {"t": "", "s": False}, "last_ts": ts,
                            "tainted": False, "fb": None, "tools": 0})
        return

    if typ == "assistant":
        if msg.get("model") and msg["model"] != "<synthetic>" and msg["model"] not in st["models"]:
            st["models"].append(msg["model"])
        u = msg.get("usage")
        if u:
            mid = msg.get("id") or o.get("uuid")
            prev = st["usage"].get(mid)
            out = max(prev[1] if prev else 0, u.get("output_tokens") or 0)
            st["usage"][mid] = [u.get("input_tokens") or 0, out, u.get("cache_creation_input_tokens") or 0, u.get("cache_read_input_tokens") or 0]
        if o.get("isApiErrorMessage"):
            st["signals"]["api_error"] = st["signals"].get("api_error", 0) + 1
        turn = _cur_turn(st)
        for b in msg.get("content") or []:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text" and turn is not None:
                txt = b.get("text") or ""
                if _TURN_LIMIT.search(txt):
                    st["signals"]["turn_limit"] = st["signals"].get("turn_limit", 0) + 1
                if o.get("isApiErrorMessage") or (len(txt) < 300 and _API_ERR.search(txt) and txt.lstrip().lower().startswith("api error")):
                    continue
                r = _scrub(st, txt, MAX_RESP)
                cur = turn["resp"]
                joined = (cur["t"] + "\n\n" + r["t"]).strip() if cur["t"] else r["t"]
                if len(joined) > MAX_RESP:
                    joined = joined[-MAX_RESP:]
                turn["resp"] = {"t": joined, "s": cur["s"] or r["s"]}
                turn["last_ts"] = ts or turn["last_ts"]
            elif bt == "tool_use":
                st["n_tool_calls"] += 1
                name = b.get("name")
                inp = b.get("input") or {}
                if touches_sensitive_path(_path_fields(inp)):
                    st["tainted_tools"] += 1
                    if turn is not None:
                        turn["tainted"] = True
                    if name not in DISPATCH_TOOLS and name != "SendMessage":
                        continue
                if turn is not None:
                    turn["resp"] = {"t": "", "s": False}  # the final answer is the text AFTER the last tool call
                    turn["tools"] = turn.get("tools", 0) + 1
                    turn["last_ts"] = ts or turn["last_ts"]
                if name in DISPATCH_TOOLS or name == "SendMessage":
                    prompt = str(inp.get("prompt") or inp.get("message") or inp.get("content") or "")
                    st["dispatches"][b.get("id")] = {
                        "ts": ts, "tool": name, "subagent_type": inp.get("subagent_type"),
                        "description": _scrub(st, str(inp.get("description") or ""), 300)["t"],
                        "prompt": _scrub(st, prompt, MAX_PROMPT),
                        "to": inp.get("to") or inp.get("agent_id") or inp.get("name"),
                        "leader_model": msg.get("model"), "cwd": o.get("cwd") or st["cwd"],
                    }
                elif name == "SubagentHandback":
                    st["handback"] = _scrub(st, str(inp.get("message") or ""), MAX_RESP)
        return

    if typ == "user":
        c = msg.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    if tid in st["dispatches"]:
                        txt = _text_of(b.get("content")) if not isinstance(b.get("content"), str) else b["content"]
                        if txt.startswith("Async agent launched"):
                            st["results"][tid] = {"ts": ts, "head": None, "async": True, "error": False}
                        else:
                            st["results"][tid] = {"ts": ts, "head": _scrub(st, txt, MAX_RESULT_HEAD)["t"], "async": False,
                                                  "error": bool(b.get("is_error"))}
                    if b.get("is_error"):
                        st["signals"]["tool_error"] = st["signals"].get("tool_error", 0) + 1


def _head_hash(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(4096)).hexdigest()[:16]


def state_path(files_dir: Path, rel: str) -> Path:
    return files_dir / f"{sha16(rel)}.json"


def update_file(path: Path, rel: str, files_dir: Path) -> tuple[dict, int, bool, dict]:
    """Fold new complete lines of ``path`` into its state. Returns (state, bytes_read, changed, scrub_hits_delta)."""
    sp = state_path(files_dir, rel)
    st = read_json(sp, None)
    if not st or st.get("v") != STATE_VERSION or st.get("rel") != rel:
        st = _new_state(rel)
    stat = path.stat()
    cur = st["cursor"]
    head = _head_hash(path) if stat.st_size >= 4096 else ""
    if stat.st_size < cur["offset"] or (cur["head"] and head != cur["head"]):
        st = _new_state(rel)  # truncated or replaced: re-read from scratch
        cur = st["cursor"]
    before = dict(st["scrub_hits"])
    if st["agent_id"] and st["meta"] is None:
        mp = path.with_name(path.name.removesuffix(".jsonl") + ".meta.json")
        meta = read_json(mp, None)
        if meta:
            st["meta"] = {"agentType": meta.get("agentType"), "toolUseId": meta.get("toolUseId"), "parentAgentId": meta.get("parentAgentId"),
                          "spawnDepth": meta.get("spawnDepth"), "description": scrub(str(meta.get("description") or "")).text[:300]}
    if stat.st_size == cur["offset"] and stat.st_mtime_ns == cur["mtime_ns"]:
        return st, 0, False, {}
    read = 0
    with open(path, "rb") as f:
        f.seek(cur["offset"])
        data = f.read()
    end = data.rfind(b"\n")
    if end >= 0:
        chunk = data[: end + 1]
        for raw in chunk.split(b"\n"):
            if not raw.strip():
                continue
            try:
                o = json.loads(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, ValueError):
                st["signals"]["bad_line"] = st["signals"].get("bad_line", 0) + 1
                continue
            if isinstance(o, dict):
                apply_record(st, o)
        read = len(chunk)
        cur["offset"] += read
    cur["size"] = stat.st_size
    cur["mtime_ns"] = stat.st_mtime_ns
    if not cur["head"] and head and cur["offset"] >= 4096:
        cur["head"] = head  # only a FULL 4 KiB head is recorded, so a growing small file never looks "replaced"
    atomic_write_json(sp, st)
    delta = {k: v - before.get(k, 0) for k, v in st["scrub_hits"].items() if v - before.get(k, 0)}
    return st, read, True, delta


def load_all_states(files_dir: Path) -> list[dict]:
    out = []
    if files_dir.exists():
        for p in sorted(files_dir.glob("*.json")):
            st = read_json(p, None)
            if st and st.get("v") == STATE_VERSION:
                out.append(st)
    return out
