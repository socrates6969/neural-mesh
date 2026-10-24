"""Read-only collectors for the mesh-ops dashboard and alerts.

Every collector tolerates missing or half-written files: it records what it could not read in `errors`
and returns the rest. Nothing here writes to a data source; workspace.db is opened with mode=ro.

Roots (override with env vars or the `over` argument, used by the tests):
  OPS_NM_ROOT   neural-mesh checkout          (default C:\\Users\\mariu\\neural-mesh)
  OPS_AC_ROOT   agent-catalog training runs   (default C:\\Users\\mariu\\ruv\\hive\\runs\\agent-catalog)
  OPS_OUT_DIR   dashboard output directory    (default C:\\Users\\mariu\\ruv\\hive\\runs\\dashboard)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path

DEFAULTS = {
    "nm": r"C:\Users\mariu\neural-mesh",
    "ac": r"C:\Users\mariu\ruv\hive\runs\agent-catalog",
    "out": r"C:\Users\mariu\ruv\hive\runs\dashboard",
}
ENV = {"nm": "OPS_NM_ROOT", "ac": "OPS_AC_ROOT", "out": "OPS_OUT_DIR"}

ROUTER_STALE_H = 48
INGEST_STALE_H = 2
NIGHTLY_STALE_H = 36

QUOTA_RE = re.compile(r"QUOTA:")
STAGE_RE = re.compile(r"^== stage \S+ done|round \d+ done, spent")
TRACE_RE = re.compile(r"Traceback \(most recent call last\)")
START_RE = re.compile(r"start: (\d+) agents, rounds (\d+)\.\.(\d+), budget \$([\d.]+), spent so far \$([\d.]+)")
SPENT_RE = re.compile(r"round \d+ done, spent \$([\d.]+)")
FED_RE = re.compile(r"router: (\d+) outcomes appended and fed")
RATE_RE = re.compile(r"(?:win rate|lower bound) ([\d.]+)")


def roots(over: dict | None = None) -> dict:
    r = {k: Path(os.environ.get(ENV[k], v)) for k, v in DEFAULTS.items()}
    for k, v in (over or {}).items():
        r[k] = Path(v)
    return r


def _iso(ts: float | None) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) if ts else None


def _age_h(ts: float | None) -> float | None:
    return (time.time() - ts) / 3600 if ts else None


def _stat(p: Path) -> dict | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime": st.st_mtime, "mtime_iso": _iso(st.st_mtime)}


def _json(p: Path, errors: list):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        errors.append("%s: %s" % (p, e))
        return None


def _jsonl(p: Path, errors: list, tail: int | None = None) -> list[dict]:
    """Parse a jsonl file line by line; a torn or bad line is skipped, not fatal."""
    rows: list[dict] = []
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return rows
    except OSError as e:
        errors.append("%s: %s" % (p, e))
        return rows
    for line in lines[-tail:] if tail else lines:
        line = line.strip()
        if line:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _count_lines(p: Path) -> int:
    try:
        with open(p, "rb") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _tail_text(p: Path, nbytes: int = 65536) -> str:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def scan_log(p: Path) -> dict:
    """Pattern matches in a training log. These logs are small (<1 MB), so the whole file is read."""
    st = _stat(p)
    out = {"path": str(p), "exists": st is not None, "size": st["size"] if st else 0,
           "mtime": st["mtime"] if st else None, "quota": [], "stage": [], "traceback": [], "start": None,
           "spent": None, "fed": 0, "last": None}
    if not st:
        return out
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if QUOTA_RE.search(ln):
            out["quota"].append(ln)
        if STAGE_RE.search(ln):
            out["stage"].append(ln)
        if TRACE_RE.search(ln):
            out["traceback"].append(ln)
        m = START_RE.search(ln)
        if m:
            out["start"] = {"agents": int(m[1]), "rounds": [int(m[2]), int(m[3])], "budget": float(m[4]),
                            "spent_at_start": float(m[5])}
        m = SPENT_RE.search(ln)
        if m:
            out["spent"] = float(m[1])
        m = FED_RE.search(ln)
        if m:
            out["fed"] += int(m[1])
    out["last"] = lines[-1] if lines else None
    return out


# ---------------------------------------------------------------- training

def training(ac: Path) -> dict:
    errors: list[str] = []
    train = ac / "train"
    batches = []
    if train.is_dir():
        for b in sorted(train.iterdir()):
            if b.is_dir() and "archive" not in b.name:
                batches.append(_batch(ac, b, errors))
    logs = {}
    paths = sorted(ac.glob("*_stdout.log")) + (sorted(train.glob("*/train.log")) if train.is_dir() else [])
    for p in paths:
        if "archive" not in str(p):
            logs[str(p)] = scan_log(p)
    return {"root": str(ac), "exists": ac.is_dir(), "batches": batches, "logs": logs, "errors": errors}


def _batch(ac: Path, b: Path, errors: list) -> dict:
    lb = _json(b / "leaderboard.json", errors) or {}
    agents = []
    for field, rows in (lb.get("fields") or {}).items():
        for r in rows or []:
            agents.append({"agent": r.get("agent"), "field": field, "held_base": r.get("held_base"),
                           "held_with_pack": r.get("held_with_pack"), "pack_version": r.get("pack_version"),
                           "status": r.get("pack_status"), "pooled": r.get("pooled") or {}})
    packs = {}
    pdir = b / "packs"
    for st_path in sorted(pdir.glob("*/*/status.json")) if pdir.is_dir() else []:
        st = _json(st_path, errors)
        if not isinstance(st, dict):
            continue
        rounds = st.get("rounds") or []
        packs["%s/%s" % (st_path.parent.parent.name, st_path.parent.name)] = {
            "status": st.get("status"), "version": st.get("version"), "rounds": len(rounds),
            "kept": sum(1 for r in rounds if isinstance(r, dict) and r.get("kept")),
            "pooled": st.get("pooled") or {}, "mtime": (_stat(st_path) or {}).get("mtime_iso")}
    # status.json is written every round; the leaderboard only at round end, so it can lag behind
    on_board = {}
    for a in agents:
        on_board[(a["agent"] or "").replace(":", "_") + "/" + a["field"]] = a
    for key, p in packs.items():
        if key in on_board:
            on_board[key]["status"] = p["status"]
            continue
        name, field = key.rsplit("/", 1)
        agents.append({"agent": name, "field": field, "held_base": None, "held_with_pack": None,
                       "pack_version": p["version"], "status": p["status"], "pooled": p["pooled"]})
    res_path = b / "results.jsonl"
    rows = _jsonl(res_path, errors)
    res_st = _stat(res_path)
    log = scan_log(ac / ("%s_stdout.log" % b.name))
    spent = lb.get("spent_usd")
    if spent is None:
        spent = log["spent"] if log["spent"] is not None else round(sum(r.get("cost") or 0 for r in rows), 2)
    return {"name": b.name, "spent": spent, "budget": (log["start"] or {}).get("budget"), "agents": agents,
            "packs": packs, "results": len(rows), "result_errors": sum(1 for r in rows if r.get("error")),
            "held_rows": sum(1 for r in rows if str(r.get("phase") or "").startswith("held")),
            "results_mtime": res_st["mtime_iso"] if res_st else None,
            "results_age_h": _age_h(res_st["mtime"]) if res_st else None,
            "last_log": log["last"], "quota_stops": len(log["quota"]), "fed_router": log["fed"]}


# ---------------------------------------------------------------- live mesh

def mesh(nm: Path) -> dict:
    errors: list[str] = []
    L = nm / "data" / "lifelong"
    M = L / "mesh"
    out_tail = _jsonl(L / "outcomes.jsonl", errors, tail=20)
    rs = _stat(M / "router_state.json")
    ingest_rows = _jsonl(L / "runs.jsonl", errors)
    exp_rows = _jsonl(M / "experience.jsonl", errors)
    fields: dict[str, int] = {}
    for r in exp_rows:
        k = r.get("field") or "?"
        fields[k] = fields.get(k, 0) + 1
    return {
        "router": {"state": rs, "age_h": _age_h(rs["mtime"]) if rs else None,
                   "stale": rs is None or _age_h(rs["mtime"]) > ROUTER_STALE_H,
                   "outcomes": _count_lines(L / "outcomes.jsonl"),
                   "last_outcome": out_tail[-1].get("ts") if out_tail else None,
                   "last_outcome_project": out_tail[-1].get("project") if out_tail else None,
                   "fed_by_ingest": sum(((r.get("mesh") or {}).get("fed") or 0) for r in ingest_rows)},
        "promotions": _jsonl(M / "promotions.jsonl", errors),
        "experience": {"n": len(exp_rows), "fields": fields, "stat": _stat(M / "experience.jsonl"),
                       "recent": [{"task": r.get("task"), "agent": r.get("agent"), "field": r.get("field")}
                                  for r in exp_rows[-6:]][::-1]},
        "workspace": workspace(M / "workspace.db", errors),
        "ingest": ingest(L, ingest_rows),
        "errors": errors,
    }


def workspace(db: Path, errors: list) -> dict:
    st = _stat(db)
    out = {"exists": st is not None, "stat": st, "entries": 0, "verified": 0, "max_verified_id": 0,
           "kinds": {}, "recent": [], "error": None}
    if not st:
        return out
    try:
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True, timeout=2)
        try:
            con.row_factory = sqlite3.Row
            out["entries"] = con.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            out["verified"] = con.execute("SELECT COUNT(*) FROM entries WHERE verified=1").fetchone()[0]
            out["max_verified_id"] = con.execute("SELECT COALESCE(MAX(id),0) FROM entries WHERE verified=1").fetchone()[0]
            out["kinds"] = {r[0]: r[1] for r in con.execute("SELECT kind, COUNT(*) FROM entries GROUP BY kind")}
            for r in con.execute("SELECT id, task_id, agent, kind, field, enc, verified, confidence, created, "
                                 "substr(content,1,160) AS c FROM entries ORDER BY id DESC LIMIT 8"):
                out["recent"].append({"id": r["id"], "task": r["task_id"], "agent": r["agent"], "kind": r["kind"],
                                      "field": r["field"], "verified": bool(r["verified"]),
                                      "confidence": r["confidence"], "created": _iso(r["created"]),
                                      "content": "(encrypted)" if r["enc"] else r["c"]})
        finally:
            con.close()
    except sqlite3.Error as e:
        out["error"] = str(e)
        errors.append("workspace.db: %s" % e)
    return out


def ingest(L: Path, rows: list[dict]) -> dict:
    last = rows[-1] if rows else None
    exits = re.findall(r"^(\S+) ingest run exit (-?\d+)", _tail_text(L / "ingest-runs.log", 8192), re.M)
    age_h = None
    if last and last.get("ts"):
        try:
            age_h = _age_h(datetime.fromisoformat(last["ts"]).timestamp())
        except ValueError:
            pass
    return {"runs": len(rows), "last_ts": last.get("ts") if last else None, "age_h": age_h,
            "totals": (last or {}).get("totals"), "runtime_s": (last or {}).get("runtime_s"),
            "judge": (last or {}).get("judge"),
            "last_exit": int(exits[-1][1]) if exits else None, "last_exit_ts": exits[-1][0] if exits else None,
            "failed_recent": sum(1 for _, code in exits[-12:] if code != "0")}


# ---------------------------------------------------------------- nightly LLM

def nightly(nm: Path) -> dict:
    errors: list[str] = []
    D = nm / "llm" / "models"
    runs = _jsonl(D / "runs.jsonl", errors)
    state = _json(D / "state.json", errors) or {}
    lock = None
    try:
        parts = (D / "nightly.lock").read_text().split()
        lock = {"pid": parts[0], "since": _iso(float(parts[1])) if len(parts) > 1 else None}
    except (OSError, ValueError, IndexError):
        pass
    tail = [ln.strip() for ln in re.split(r"[\r\n]+", _tail_text(D / "nightly.log", 32768))
            if ln.strip() and "it/s]" not in ln and "s/it]" not in ln]
    return {"runs": [_run(r) for r in runs[-7:]][::-1], "n_runs": len(runs),
            "production": state.get("production"), "last_nightly": state.get("last_nightly"),
            "trained": len(state.get("trained_ids") or []), "lock": lock, "log_tail": tail[-12:],
            "log_stat": _stat(D / "nightly.log"), "errors": errors}


def _run(r: dict) -> dict:
    stages = {s.get("name"): s for s in r.get("stages") or [] if isinstance(s, dict)}
    ev = stages.get("eval") or {}
    pr = stages.get("promote") or {}
    m = RATE_RE.search(ev.get("reason") or "")
    return {"run_id": r.get("run_id"), "started": r.get("started"), "ended": r.get("ended"),
            "status": r.get("status"), "exit": r.get("exit"), "synthetic": r.get("synthetic"),
            "stages": [[s.get("name"), s.get("status"), s.get("exit")] for s in stages.values()],
            "ver": (ev.get("detail") or {}).get("ver"),
            "n_new": ((stages.get("train") or {}).get("detail") or {}).get("n_new"),
            "gate": ev.get("reason"), "rate": float(m[1]) if m else None,
            "promoted": bool((pr.get("detail") or {}).get("promoted"))}


# ---------------------------------------------------------------- everything

def collect(over: dict | None = None) -> dict:
    t0 = time.time()
    r = roots(over)
    st = {"generated": _iso(t0), "generated_epoch": t0, "roots": {k: str(v) for k, v in r.items()},
          "training": training(r["ac"]), "mesh": mesh(r["nm"]), "nightly": nightly(r["nm"])}
    st["health"] = health(st)
    st["collect_s"] = round(time.time() - t0, 3)
    return st


def health(st: dict) -> list[dict]:
    """Checks shown on the dashboard, worst first. level is ok | warn | err."""
    out = []

    def add(level, name, text):
        out.append({"level": level, "name": name, "text": text})

    tr, me, ni = st["training"], st["mesh"], st["nightly"]
    if not tr["exists"]:
        add("warn", "training", "agent-catalog root not found: %s" % tr["root"])
    rt = me["router"]
    if rt["state"] is None:
        add("err", "router", "router_state.json missing")
    else:
        add("err" if rt["stale"] else "ok", "router",
            "router_state updated %.1f h ago (stale after %d h)" % (rt["age_h"], ROUTER_STALE_H))
    ig = me["ingest"]
    if ig["last_ts"] is None:
        add("warn", "ingest", "no ingest runs recorded")
    else:
        stale = ig["age_h"] is not None and ig["age_h"] > INGEST_STALE_H
        add("warn" if stale else "ok", "ingest", "last ingest %s (%.1f h ago)" % (ig["last_ts"], ig["age_h"] or 0))
    if ig["last_exit"] not in (None, 0):
        add("err", "ingest exit", "last ingest run exited %s at %s" % (ig["last_exit"], ig["last_exit_ts"]))
    elif ig["failed_recent"]:
        add("warn", "ingest failures", "%d of the last 12 ingest runs exited non-zero" % ig["failed_recent"])
    if ni["runs"]:
        last = ni["runs"][0]
        if last["exit"] not in (0, None) or last["status"] not in ("ok", None):
            add("err", "nightly", "last nightly %s status %s exit %s" % (last["run_id"], last["status"], last["exit"]))
        else:
            add("ok", "nightly", "last nightly %s ok%s" % (last["run_id"], ", promoted" if last["promoted"] else ""))
        lt = ni["log_stat"]
        if lt and _age_h(lt["mtime"]) > NIGHTLY_STALE_H:
            add("warn", "nightly log", "nightly.log not written for %.0f h" % _age_h(lt["mtime"]))
    else:
        add("warn", "nightly", "no nightly runs recorded")
    for p, lg in tr["logs"].items():
        pp = Path(p)
        name = pp.parent.name + "/" + pp.name if pp.name == "train.log" else pp.name
        if lg["quota"]:
            latest = bool(lg["last"] and QUOTA_RE.search(lg["last"]))
            add("err" if latest else "warn", "quota " + name, "%d quota stop(s), %s: %s" % (
                len(lg["quota"]), "latest line is a quota stop" if latest else "training continued after it",
                lg["quota"][-1][:160]))
        if lg["traceback"]:
            add("err", "traceback " + name, "%d Traceback(s) in log" % len(lg["traceback"]))
    for b in tr["batches"]:
        if b["result_errors"]:
            add("warn", "results " + b["name"], "%d result rows carry an error" % b["result_errors"])
        if b["budget"] and b["spent"] is not None and b["spent"] >= b["budget"]:
            add("warn", "budget " + b["name"], "spent $%.2f of $%.0f budget" % (b["spent"], b["budget"]))
    if me["workspace"]["error"]:
        add("warn", "workspace", "workspace.db unreadable: %s" % me["workspace"]["error"])
    for e in tr["errors"] + me["errors"] + ni["errors"]:
        add("warn", "read error", e[:200])
    order = {"err": 0, "warn": 1, "ok": 2}
    return sorted(out, key=lambda h: order[h["level"]])