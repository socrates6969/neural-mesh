"""Incremental, resumable reader of Claude Code transcripts.

Every transcript file (session main file or subagent sidechain) has its own byte cursor and its own compact, SCRUBBED
per-file state under ``state/files/<sha16(rel)>.json``. A run reads only the complete new lines (up to the last newline)
since the cursor, folds them into the file state, and saves state + cursor atomically. A file that shrank or whose head
changed is re-read from byte 0 with a fresh state (outputs are de-duplicated by id, so this is safe).
Nothing raw is stored: tool inputs and tool results are never kept, except the scrubbed head of Agent results. Every tool call
leaves a content-free call record (trajectory.py: tool, argument shape, shell verbs, file extension, ok/err, exit code, test counts,
per-call taint class) plus the scrubbed assistant text that preceded it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter
from pathlib import Path

from . import trajectory as tj
from .scrub import scrub, touches_sensitive_path
from .util import atomic_write_json, read_json, sha16

STATE_VERSION = 4   # v2: per-call trajectory records (calls / pending) in every turn; v3: shell code writes, background results
MAX_PROMPT = 6000
MAX_RESP = 10000
MAX_RESULT_HEAD = 600
WITHHELD = "[withheld: written after this agent read a sensitive file]"   # placeholder, flagged secret, so "a report existed" survives
DISPATCH_TOOLS = {"Agent", "Task"}
_PATH_KEYS = ("file_path", "path", "notebook_path", "pattern", "glob", "command", "url", "cwd", "directory", "dir", "paths", "files")
_SYSTEM_PREFIX = re.compile(r"^\s*(?:<(?:task-notification|command-name|command-message|command-args|local-command|system-reminder|bash-|user-prompt-submit-hook)|Caveat:|\[Request interrupted)")
_SYS_REMINDER = re.compile(r"<system-reminder>[\s\S]*?</system-reminder>")
_API_ERR = re.compile(r"(?i)API Error|rate limit|weekly limit|usage limit|overloaded|\b429\b")
_TURN_LIMIT = re.compile(r"(?i)maximum number of turns|max[_ ]turns|turn limit")


def classify_rel(rel: str) -> str | None:
    """'main' (<project>/<session>.jsonl), 'agent' (any agent-*.jsonl at ANY depth below <project>/<session>/), else None.
    Any nesting under a session dir is accepted, so a new layout (nested teams, workflows) is not silently dropped."""
    parts = rel.split("/")
    if len(parts) == 2:
        return "main"
    if len(parts) >= 3 and parts[-1].startswith("agent-"):
        return "agent"
    return None


def discover_report(projects: Path) -> tuple[list[tuple[Path, str]], dict]:
    """(transcript files, skipped counts by reason). Only counts are reported, never paths. Within a session, subagent files come
    BEFORE the main file, so a secret value seen in a subagent's sensitive read is registered before the leader's echo is read."""
    out, skipped = [], {}
    if not projects.exists():
        return out, skipped
    for p in sorted(projects.rglob("*.jsonl")):
        rel = p.relative_to(projects).as_posix()
        if classify_rel(rel):
            out.append((p, rel))
        else:
            why = "top_level" if rel.count("/") == 0 else "not_agent_file"
            skipped[why] = skipped.get(why, 0) + 1
    out.sort(key=lambda x: (x[1].split("/")[0], x[1].split("/")[1].removesuffix(".jsonl"), classify_rel(x[1]) == "main", x[1]))
    return out, skipped


def discover(projects: Path) -> list[tuple[Path, str]]:
    """All transcript files with their path relative to ``projects`` (sorted, deterministic)."""
    return discover_report(projects)[0]


def _new_state(rel: str) -> dict:
    parts = rel.split("/")
    agent_id = None
    if len(parts) >= 3:
        m = re.match(r"agent-([A-Za-z0-9_-]+)\.jsonl$", parts[-1])
        agent_id = m.group(1) if m else parts[-1]
    return {
        "v": STATE_VERSION, "rel": rel, "project_dir": parts[0], "session": parts[1].removesuffix(".jsonl"),
        "agent_id": agent_id, "meta": None,
        "cursor": {"offset": 0, "size": 0, "mtime_ns": 0, "head": ""},
        "cwd": None, "first_ts": None, "last_ts": None,
        "dispatches": {}, "results": {}, "usage": {}, "models": [],
        "turns": [], "handback": None, "signals": {}, "tainted_tools": 0, "n_tool_calls": 0,
        "scrub_hits": {}, "pending": {}, "orphan_calls": 0,
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


class SecretRegistry:
    """Per-session set of HMAC-64 fingerprints of VALUES seen in the results of sensitive-file reads (KEY=VALUE lines and long
    mixed tokens). Any later text of the same session (subagent or leader) containing such a value is redacted and flagged secret,
    so a plain-word echo ("the db password is hunter2x") cannot slip past the regex scrubber. Only keyed, truncated fingerprints
    are stored (key file next to the state), never the values."""
    _TOK = re.compile(r"[^\s\"'`,;:=()\[\]{}<>|]+")
    _KV = re.compile(r"(?mi)^[\s\"']*(?:export\s+)?[\w.-]*(?:pass|pwd|secret|token|key|credential|auth|private|salt|dsn|cookie|session)[\w.-]*"
                     r"[\"']?\s*[:=]\s*[\"']?([^\s\"'#]{6,})")
    FILE_DUMP_VERBS = {"cat", "type", "get-content", "gc", "head", "tail", "more", "less", "bat", "strings", "xxd", "od"}

    def __init__(self, path: Path | None):
        self.path = path
        self.by: dict[str, set] = {}
        self.dirty = False
        self.key = b""
        if path is not None:
            kp = path.with_name("secret_hmac.key")
            if kp.exists():
                self.key = bytes.fromhex(kp.read_text(encoding="ascii").strip())
            else:
                kp.parent.mkdir(parents=True, exist_ok=True)
                self.key = os.urandom(32)
                kp.write_text(self.key.hex(), encoding="ascii")
            self.by = {k: set(v) for k, v in read_json(path, {}).items()}

    def fp(self, tok: str) -> str:
        return hmac.new(self.key, tok.encode("utf-8", "replace"), "sha256").hexdigest()[:16]

    def add_from_result(self, session: str, text: str) -> int:
        vals = set(m.group(1).rstrip(".") for m in self._KV.finditer(text or ""))
        vals |= {t for t in self._TOK.findall(text or "") if len(t) >= 16 and sum((any(c.isdigit() for c in t), any(c.isupper() for c in t),
                                                                                     any(c.islower() for c in t), any(not c.isalnum() for c in t))) >= 3}
        fps = {self.fp(v) for v in vals if len(v) >= 6}
        if fps - self.by.get(session, set()):
            self.by.setdefault(session, set()).update(fps)
            self.dirty = True
        return len(fps)

    def redact(self, session: str, text: str) -> tuple[str, int]:
        known = self.by.get(session)
        if not known or not text:
            return text, 0
        n = 0

        def sub(m: re.Match) -> str:
            nonlocal n
            t = m.group(0)
            for cand in (t, t.rstrip(".")):
                if len(cand) >= 6 and self.fp(cand) in known:
                    n += 1
                    return "[REDACTED:session_secret]"
            return t
        return self._TOK.sub(sub, text), n

    def save(self) -> None:
        if self.path is not None and self.dirty:
            atomic_write_json(self.path, {k: sorted(v) for k, v in self.by.items()})
            self.dirty = False


_REG = SecretRegistry(None)   # set by update_file for the duration of a file


def _scrub(st: dict, text: str, limit: int) -> dict:
    """Scrub and cap a text; keep the secret flag so sft/memory can drop it."""
    r = scrub(text[: limit * 2] if text else "")
    _hits(st, r.hits)
    t, n = _REG.redact(st.get("session") or "", r.text)
    if n:
        st["scrub_hits"]["session_secret"] = st["scrub_hits"].get("session_secret", 0) + n
        r = type(r)(t, r.hits.copy())
        r.hits["session_secret"] += n
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
                            "tainted": False, "fb": None, "tools": 0, "calls": []})
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
                if turn.get("ct"):          # this turn read a sensitive file's content: nothing it says afterwards is stored
                    turn["resp"] = {"t": WITHHELD, "s": True}
                    st["withheld"] = st.get("withheld", 0) + 1
                    continue
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
                if touches_sensitive_path(_path_fields(inp)) and name not in DISPATCH_TOOLS and name != "SendMessage" and turn is not None:
                    turn["ct"] = True
                _record_call(st, turn, b, name, inp, ts)
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
                    st["handback"] = {"t": WITHHELD, "s": True} if (turn or {}).get("ct") else _scrub(st, str(inp.get("message") or ""), MAX_RESP)
        return

    if typ == "user":
        c = msg.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tid = b.get("tool_use_id")
                    _record_result(st, tid, b)
                    if tid in st["dispatches"]:
                        txt = _text_of(b.get("content")) if not isinstance(b.get("content"), str) else b["content"]
                        if txt.startswith("Async agent launched"):
                            st["results"][tid] = {"ts": ts, "head": None, "async": True, "error": False}
                        else:
                            st["results"][tid] = {"ts": ts, "head": _scrub(st, txt, MAX_RESULT_HEAD)["t"], "async": False,
                                                  "error": bool(b.get("is_error"))}
                    if b.get("is_error"):
                        st["signals"]["tool_error"] = st["signals"].get("tool_error", 0) + 1


def _record_call(st: dict, turn: dict | None, b: dict, name, inp, ts) -> None:
    """Content-free call record (trajectory.py) appended to the current turn; the preceding assistant text is its rationale."""
    if turn is None:
        st["orphan_calls"] = st.get("orphan_calls", 0) + 1
        return
    n = tj._norm_name(name)
    verbs = tj.shell_verbs(str(inp.get("command") or "")) if n in tj.SHELL_TOOLS and isinstance(inp, dict) else []
    rec = {"n": n, "k": tj.kind_of(n), "sh": tj.arg_shape(inp), "ts": ts, "ok": None}
    if verbs:
        rec["v"] = verbs
        ck = tj.check_kind(verbs)
        if ck:
            rec["c"] = ck
        cmd = str(inp.get("command") or "")
        wp = tj.write_pos(cmd)
        if wp >= 0:
            rec["we"] = True
            if ck and wp > tj.check_pos(cmd):
                rec["wa"] = True     # the code write comes AFTER the test/build inside the same command
    x = tj.file_ext(inp)
    if x:
        rec["x"] = x
    tt = tj.call_taint(n, inp, _path_fields(inp), verbs)
    if tt:
        rec["tt"] = tt
        if tt == "write" or (tt == "read" and n not in tj.READ_TOOLS):
            turn["ct"] = True       # content taint now (a failed Read is cleared only when its result arrives)
    why = (turn.get("resp") or {}).get("t") or ""
    if why and not tt:
        rec["why"] = {"t": why[-tj.MAX_WHY:], "s": bool((turn.get("resp") or {}).get("s"))}
    calls = turn.setdefault("calls", [])
    calls.append(rec)
    if b.get("id"):
        st.setdefault("pending", {})[b["id"]] = [len(st["turns"]) - 1, len(calls) - 1]


def _record_result(st: dict, tid, b: dict) -> None:
    pos = st.setdefault("pending", {}).pop(tid, None)
    if not pos:
        return
    try:
        rec = st["turns"][pos[0]]["calls"][pos[1]]
    except (IndexError, KeyError):
        return
    c = b.get("content")
    txt = c if isinstance(c, str) else _text_of(c)
    rec.update(tj.result_summary(txt, bool(b.get("is_error"))))
    if rec.get("tt") == "read":
        if rec["n"] in tj.READ_TOOLS and rec["ok"] is False:
            return                  # "File does not exist" etc.: no content reached the context
        st["turns"][pos[0]]["ct"] = True
        verbs = rec.get("v") or []
        if rec["n"] in tj.READ_TOOLS or (verbs and verbs[0].split()[0] in SecretRegistry.FILE_DUMP_VERBS):
            _REG.add_from_result(st.get("session") or "", txt)      # values of a dumped secret file (not of grep/build output)


def _head_hash(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read(4096)).hexdigest()[:16]


def state_path(files_dir: Path, rel: str) -> Path:
    return files_dir / f"{sha16(rel)}.json"


def update_file(path: Path, rel: str, files_dir: Path, registry: SecretRegistry | None = None) -> tuple[dict, int, bool, dict]:
    """Fold new complete lines of ``path`` into its state. Returns (state, bytes_read, changed, scrub_hits_delta)."""
    global _REG
    _REG = registry or SecretRegistry(None)
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
