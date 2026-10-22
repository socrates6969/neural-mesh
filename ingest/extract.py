"""Assemble per-file states into sessions and extract outcomes, sft candidates and memory chunks.

Outcome labels are HEURISTIC (a proxy): they come from text cues in the dispatch and the agent's final report plus
structural signals (no final report, turn limit, API errors, a later re-dispatch with the same description, a
continuation). Agreement with the evaluator's 55 hand labels is measured and reported, not assumed.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import trajectory as tj

TAINT_MODE = os.environ.get("INGEST_TAINT", "agent")   # merged opt-in: "call" to enable;      # "call" (gap D: per tool call) | "agent" (legacy: any sensitive mention taints all)

OUTCOME_REWARD = {"success": 1.0, "fix-round": 0.6, "defect": 0.4, "blocked": 0.2, "re-dispatch": 0.1}

_FIX_TASK = re.compile(r"(?i)\bfix(?:es|ing)?[- ]?rounds?\b|\bround \d+\b.*\bfix|\bfix(?:es|ing)?\b.{0,60}\b(?:defects?|D\d+|findings?|CI|failures?|failing|review|regression|bugs?)\b"
                       r"|\baddress(?:ing)? (?:the )?(?:review|findings|defects|comments)\b|\bCI (?:fail|red|follow[- ]?up|round)\b|\bre-?fix\b")
_DEFECT = re.compile(r"\bTEST_BLOCK\b|\bPASS WITH NOTES\b|\bREJECT(?:ED)?\b|\bCHANGES[_ ]REQUESTED\b|\bBLOCKING\b|\bdefects? (?:found|D\d)"
                     r"|\b[1-9]\d* (?:tests? )?(?:failed|failing|failures?)\b|\bFAIL(?:ED|S)?\b(?! ?(?:=|:)? ?0\b)|\bneeds? (?:fixes|fixing|a fix)\b")
_BLOCKED = re.compile(r"(?i)\bblocked (?:by|on)\b|\bpermission (?:denied|prompt)\b|\bcannot proceed\b|\bawaiting (?:user|approval|your)\b"
                      r"|\bneed(?:s)? your (?:approval|go-ahead|decision)\b|\bstopp(?:ed|ing) (?:and|to) report\b|\bhook blocked\b|\bwas blocked\b|\bdenied by\b")
_REDISPATCH = re.compile(r"(?i)\bturn limit\b|\bmax(?:imum)? (?:number of )?turns\b|\brate.?limit|\bweekly limit\b|\busage limit\b|\b429\b|\boverloaded\b"
                         r"|\bworktree\b.{0,40}\b(?:block|isolat|fail)|\bran out of (?:turns|context)\b|\bcontext (?:limit|window) (?:reached|exceeded)\b")
_RETRY_WORDS = re.compile(r"(?i)\b(?:retry|resume|continue|again|attempt \d|re-?dispatch|re-?run|second try)\b")
_NEG_FEEDBACK = re.compile(r"(?i)^\s*(?:no\b|nope\b|wrong\b|that'?s (?:wrong|not)|doesn'?t work|does not work|it'?s still|still (?:broken|failing|not)"
                           r"|why did you|stop\b|undo\b|revert\b|you (?:broke|didn'?t)|that is not)")

STOP = set(("a an and are as at be been but by can do does done for from has have he her his i if in into is it its me my no not of on or our "
            "same so than that the their them then there these they this to up us was we were what when where which while who will with you "
            "your yours just only also any all after before now new get got one two per via vs here very more most other own each both should "
            "would could must may might never always please run step").split())


def features(text: str, k: int = 24) -> list[str]:
    toks = [t for t in re.split(r"[^a-z0-9]+", str(text).lower()) if len(t) >= 2 and not t.isdigit() and t not in STOP]
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def parse_ts(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def project_of(cwd: str | None, project_dir: str) -> str:
    if cwd:
        name = re.split(r"[\\/]", cwd.rstrip("\\/"))[-1]
        if name:
            return name
    return project_dir


def _norm_desc(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


def classify(task: str, report: str, result_head: str, signals: dict, later: dict) -> tuple[str, list[str]]:
    """Heuristic outcome. Precedence: re-dispatch > blocked > fix-round > defect > success."""
    sig = []
    if later.get("no_report"):
        sig.append("no_final_report")
    if signals.get("turn_limit"):
        sig.append("turn_limit")
    if signals.get("api_error"):
        sig.append("api_error")
    if later.get("redispatched"):
        sig.append("redispatched_later")
    if later.get("continued"):
        sig.append("continued")
    tail = (report or "")[-700:] + "\n" + (result_head or "")
    if _REDISPATCH.search(tail):
        sig.append("redispatch_cue")
    if _BLOCKED.search(tail):
        sig.append("blocked_cue")
    if _FIX_TASK.search((task or "")[:300]):
        sig.append("fix_task")
    if _DEFECT.search(report or ""):
        sig.append("defect_cue")
    if later.get("result_error"):
        sig.append("result_error")
    s = set(sig)
    if s & {"no_final_report", "turn_limit", "redispatched_later", "redispatch_cue", "result_error", "api_error"}:
        return "re-dispatch", sig
    if "blocked_cue" in s:
        return "blocked", sig
    if "fix_task" in s:
        return "fix-round", sig
    if "defect_cue" in s:
        return "defect", sig
    return "success", sig


@dataclass
class DispatchItem:
    id: str
    ts: str
    end_ts: str | None
    project: str
    session: str
    kind: str
    agent: str
    model: str | None
    spawn_depth: int | None
    description: str
    prompt: str
    prompt_secret: bool
    report: str
    report_secret: bool
    tainted: bool
    result_head: str
    signals: dict
    later: dict
    tokens: dict | None
    settled: bool
    chain: str = ""      # id of the dispatch this item belongs to (a SendMessage continuation shares its origin's chain)
    calls: list = field(default_factory=list)     # content-free call records of the agent's whole trajectory (trajectory.py)
    ev: dict = field(default_factory=dict)        # execution evidence (trajectory.evidence) + optional later verdict


@dataclass
class TurnItem:
    id: str
    ts: str
    end_ts: str
    project: str
    session: str
    model: str | None
    user: str
    user_secret: bool
    resp: str
    resp_secret: bool
    tainted: bool
    feedback: str | None
    tools: int
    settled: bool
    calls: list = field(default_factory=list)
    ev: dict = field(default_factory=dict)
    agent_id: str | None = None                   # None = main thread turn; else the subagent whose turn this is
    agent: str | None = None


@dataclass
class Session:
    project_dir: str
    session: str
    main: dict | None = None
    agents: dict = field(default_factory=dict)          # toolUseId -> agent state
    agents_by_id: dict = field(default_factory=dict)    # agentId -> agent state


def group_sessions(states: list[dict]) -> list[Session]:
    by: dict[tuple, Session] = {}
    for st in states:
        key = (st["project_dir"], st["session"])
        s = by.setdefault(key, Session(*key))
        if st["agent_id"] is None:
            s.main = st
        else:
            s.agents_by_id[st["agent_id"]] = st
            tid = (st.get("meta") or {}).get("toolUseId")
            if tid:
                s.agents[tid] = st
    return [by[k] for k in sorted(by)]


def _tokens(st: dict | None) -> dict | None:
    if not st or not st.get("usage"):
        return None
    t = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
    for i, o, cc, cr in st["usage"].values():
        t["input"] += i
        t["output"] += o
        t["cache_creation"] += cc
        t["cache_read"] += cr
    t["total"] = sum(t.values())
    t["api_calls"] = len(st["usage"])
    return t


def agent_calls(st: dict | None) -> list[dict]:
    return [c for t in (st or {}).get("turns") or [] for c in t.get("calls") or []]


def is_tainted(st_or_turn_calls: list[dict], legacy: bool) -> bool:
    return legacy if TAINT_MODE == "agent" else tj.tainted(st_or_turn_calls)


def _final(agent: dict | None) -> tuple[str, bool]:
    if not agent:
        return "", False
    hb = agent.get("handback")
    if hb and hb.get("t"):
        return hb["t"], bool(hb.get("s"))
    for turn in reversed(agent.get("turns") or []):
        r = turn.get("resp") or {}
        if r.get("t"):
            return r["t"], bool(r.get("s"))
        break
    return "", False


def dispatch_items(s: Session, now: float, settle_s: float) -> list[DispatchItem]:
    files = ([s.main] if s.main else []) + list(s.agents_by_id.values())
    all_d = []
    for st in files:
        for tid, d in (st.get("dispatches") or {}).items():
            all_d.append((tid, d, st))
    all_d.sort(key=lambda x: (x[1].get("ts") or "", x[0]))
    dispatches = [(t, d, st) for t, d, st in all_d if d["tool"] in ("Agent", "Task")]
    sends = [(t, d, st) for t, d, st in all_d if d["tool"] == "SendMessage"]
    out = []
    for tid, d, owner in dispatches:
        agent = s.agents.get(tid)
        res = (owner.get("results") or {}).get(tid) or {}
        report, rsec = _final(agent)
        if not report and res.get("head") and not res.get("async"):
            report, rsec = res["head"], False
        cont = [x for x in sends if agent and x[1].get("to") == agent.get("agent_id")]
        nd = _norm_desc(d.get("description"))
        redis = False
        for t2, d2, _ in dispatches:
            if t2 == tid or (d2.get("ts") or "") <= (d.get("ts") or ""):
                continue
            if parse_ts(d2.get("ts")) - parse_ts(d.get("ts")) > 12 * 3600:
                break
            n2 = _norm_desc(d2.get("description"))
            if nd and (n2 == nd or (d2.get("subagent_type") == d.get("subagent_type") and _RETRY_WORDS.search(d2.get("description") or "")
                                    and len(set(n2.split()) & set(nd.split())) >= max(2, len(nd.split()) // 2))):
                redis = True
                break
        last_ts = (agent or {}).get("last_ts") or d.get("ts")
        settled = bool(agent and agent.get("handback")) or (bool(res) and not res.get("async") and bool(agent)) \
            or (now - parse_ts(last_ts) > settle_s)
        meta = (agent or {}).get("meta") or {}
        calls = agent_calls(agent)
        tnt = is_tainted(calls, bool(agent and agent.get("tainted_tools")))
        out.append(DispatchItem(
            id=f"out:{tid}", ts=d.get("ts"), end_ts=last_ts, project=project_of(d.get("cwd") or owner.get("cwd"), s.project_dir), session=s.session,
            kind="dispatch", agent=d.get("subagent_type") or meta.get("agentType") or "general-purpose",
            model=((agent or {}).get("models") or [None])[0], spawn_depth=meta.get("spawnDepth"),
            description=d.get("description") or "", prompt=d["prompt"]["t"], prompt_secret=bool(d["prompt"]["s"]),
            report=report, report_secret=rsec, tainted=tnt,
            result_head=res.get("head") or "", signals=dict((agent or {}).get("signals") or {}),
            later={"no_report": not report, "redispatched": redis, "continued": len(cont), "result_error": bool(res.get("error"))},
            tokens=_tokens(agent), settled=settled, chain=f"out:{tid}", calls=calls, ev=tj.evidence(calls)))
        # continuations: one item per SendMessage. Messages to a running agent are queued into its context (not separate user turns), so the
        # report of a continuation is the agent's final report when the agent kept working after the message; tokens stay on the origin.
        for t3, d3, _ in cont:
            after = parse_ts((agent or {}).get("last_ts")) > parse_ts(d3.get("ts"))
            out.append(DispatchItem(
                id=f"out:{t3}", ts=d3.get("ts"), end_ts=last_ts, project=project_of(d3.get("cwd"), s.project_dir),
                session=s.session, kind="continuation", agent=d.get("subagent_type") or meta.get("agentType") or "general-purpose",
                model=((agent or {}).get("models") or [None])[0], spawn_depth=meta.get("spawnDepth"),
                description=(d.get("description") or ""), prompt=d3["prompt"]["t"], prompt_secret=bool(d3["prompt"]["s"]),
                report=report if after else "", report_secret=rsec if after else False, tainted=tnt,
                result_head="", signals={}, later={"no_report": not (after and report), "redispatched": False, "continued": 0, "result_error": False},
                tokens=None, settled=settled, chain=f"out:{tid}"))
    return out


ALL_TURNS = os.environ.get("INGEST_ALL_TURNS", "0") != "0"   # merged opt-in: 1 to enable;   # gap A: also every subagent turn (default on), not only main-thread turns


def turn_items(s: Session, now: float, settle_s: float) -> list[TurnItem]:
    """Main-thread turns, plus (ALL_TURNS) every turn of every subagent transcript. A subagent's FIRST turn is its dispatch prompt,
    already an agent_report row, so it is skipped when the dispatch that spawned it is known."""
    out = []
    files = [(s.main, None)] if s.main else []
    if ALL_TURNS:
        dispatched = {tid for st in ([s.main] if s.main else []) + list(s.agents_by_id.values()) for tid in (st.get("dispatches") or {})}
        for aid in sorted(s.agents_by_id):
            files.append((s.agents_by_id[aid], aid))
    for st, aid in files:
        turns = st.get("turns") or []
        meta = st.get("meta") or {}
        for i, t in enumerate(turns):
            if aid is not None and i == 0 and meta.get("toolUseId") in dispatched:
                continue
            settled = i + 1 < len(turns) or now - parse_ts(t.get("last_ts")) > settle_s
            calls = t.get("calls") or []
            out.append(TurnItem(
                id=f"turn:{s.session}:{i}:{t.get('ts')}" if aid is None else f"turn:{s.session}:a{aid}:{i}:{t.get('ts')}",
                ts=t.get("ts"), end_ts=t.get("last_ts"),
                project=project_of(st.get("cwd"), s.project_dir), session=s.session, model=(st.get("models") or [None])[-1],
                user=t["user"]["t"], user_secret=bool(t["user"]["s"]), resp=t["resp"]["t"], resp_secret=bool(t["resp"]["s"]),
                tainted=is_tainted(calls, bool(t.get("tainted"))), feedback=t.get("fb"), tools=int(t.get("tools") or 0), settled=settled,
                calls=calls, ev=tj.evidence(calls), agent_id=aid, agent=meta.get("agentType") if aid else None))
    return out


def negative_feedback(fb: str | None) -> bool:
    return bool(fb) and _NEG_FEEDBACK.search(fb) is not None


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- memory chunks
_MEM_CUES = [
    ("root-cause", re.compile(r"(?i)\broot cause\b|\bcaused by\b|\bthe (?:bug|issue|problem|failure) (?:was|is)\b|\bfailed because\b|\bbecause the\b")),
    ("fix", re.compile(r"(?i)\bfix(?:ed)? (?:by|with|was)\b|\bthe fix\b|\bworkaround\b|\bresolved by\b|\bsolution:|\bfixed it\b")),
    ("decision", re.compile(r"(?i)\bdecid(?:ed|e)\b|\bwe chose\b|\bchose to\b|\bdecision\b|\btrade-?off\b|\binstead of\b|\bin favou?r of\b")),
    ("convention", re.compile(r"(?i)\balways\b|\bnever\b|\bmust\b|\bconvention\b|\brule:|\bdo not\b|\bdon'?t\b")),
    ("lesson", re.compile(r"(?i)\blesson\b|\bgotcha\b|\bcaveat\b|\bpitfall\b|\blearned\b|\bnote:|\bbeware\b|\bturns out\b")),
]
_MD = re.compile(r"^\s*(?:[-*+]|\d+[.)]|#+)\s*")


def memory_chunks(text: str, max_n: int = 8) -> list[tuple[str, str]]:
    """(tag, chunk) pairs: bullet lines / sentences that carry a decision, fix, root cause, convention or lesson."""
    out, seen = [], set()
    if not text:
        return out
    for line in text.splitlines():
        line = _MD.sub("", line).replace("**", "").replace("`", "").strip()
        if not line or "[REDACTED" in line or "[SENSITIVE_PATH]" in line:
            continue
        parts = [line] if len(line) <= 400 else re.split(r"(?<=[.!?])\s+", line)
        for p in parts:
            p = p.strip()
            if not (50 <= len(p) <= 400) or p.endswith(":") or p.count("|") >= 2:
                continue
            for tag, rx in _MEM_CUES:
                if rx.search(p):
                    key = re.sub(r"\W+", " ", p.lower()).strip()
                    if key not in seen:
                        seen.add(key)
                        out.append((tag, p))
                    break
            if len(out) >= max_n:
                return out
    return out
