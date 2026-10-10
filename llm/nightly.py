"""`python -m llm nightly`: one nightly round under ONE lock (llm/models/nightly.lock, taken by __main__; exit 75 if busy).

Stages, each recorded as a StageResult {name, status ok|noop|skipped|failed, exit, reason, started, ended, detail}:
  preflight  abandoned_sweep(): torn adapters (no meta.json) are removed; older unevaluated adapters are abandoned
             ("superseded"); a pending candidate whose parent is not the current production is abandoned ("stale parent");
             one whose eval failed twice (state.eval_attempts) is abandoned ("eval failed twice"). trained_ids never change here.
  train      skipped when a pending (non-abandoned, undecided) candidate exists - that one is resumed instead; noop when there
             is nothing new to train on.
  eval       the candidate, if it has no eval.json yet (state.eval_attempts[ver] is bumped first).
  promote    promote.run(ver) (gate unchanged; exit 3 there = failed stage).
The llm stages are skipped (not failed) on battery power. The lock mtime is refreshed before every stage so the 12 h stale
takeover cannot fire during a long round. The RunRecord {run_id, kind, stages, status, host_power, exit} is appended to
config.RUNS. Exit: 0 ok / nothing to do / skipped, 3 a stage failed (the later stages are skipped)."""
from __future__ import annotations

import json
import os
import shutil
import traceback
from datetime import datetime, timezone

from . import config

EXIT_OK, EXIT_FAILED = 0, 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def on_ac_power() -> bool | None:
    """True on AC, False on battery, None if unknown (non-Windows or the call failed): unknown counts as AC."""
    try:
        import ctypes
        from ctypes import wintypes

        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE), ("BatteryLifePercent", wintypes.BYTE),
                        ("SystemStatusFlag", wintypes.BYTE), ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

        s = SYSTEM_POWER_STATUS()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
            return None
        return {1: True, 0: False}.get(s.ACLineStatus & 0xFF)
    except (AttributeError, OSError, ImportError):
        return None


def touch_lock() -> None:
    try:
        os.utime(config.LOCK, None)
    except OSError:
        pass


def _history_by_ver(st: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for h in st.get("history", []):
        out.setdefault(h.get("ver"), []).append(h)
    return out


def _is_decided(hs: list[dict]) -> bool:
    from .promote import decided
    return decided({"history": hs}, hs[0]["ver"]) if hs else False


def _adapters(synthetic: bool) -> list:
    if not config.ADAPTERS.exists():
        return []
    return sorted((p for p in config.ADAPTERS.iterdir() if p.is_dir() and p.name.endswith("-syn") == synthetic), key=lambda p: p.name)


def abandoned_sweep(st: dict, synthetic: bool = False, log=print) -> dict:
    """mutates st (history 'abandoned' records); returns {"removed_torn": [...], "abandoned": [{ver, reason}], "pending": ver|None}."""
    prod = ((st.get("synthetic_production") if synthetic else st.get("production")) or {}).get("ver")
    attempts = st.get("eval_attempts", {})
    hist = _history_by_ver(st)
    out = {"removed_torn": [], "abandoned": [], "pending": None}

    def abandon(ver: str, reason: str) -> None:
        rec = {"ver": ver, "ts": _now(), "synthetic": synthetic, "promoted": False, "action": "abandoned", "reason": reason}
        st.setdefault("history", []).append(rec)
        hist.setdefault(ver, []).append(rec)
        out["abandoned"].append({"ver": ver, "reason": reason})
        log(f"abandoned {ver}: {reason}")

    live = []
    for d in _adapters(synthetic):
        if not (d / "meta.json").exists():
            if prod and d.name == prod:
                continue   # never touch the production adapter, even if damaged
            shutil.rmtree(d, ignore_errors=True)
            out["removed_torn"].append(d.name)
            log(f"removed torn adapter {d.name} (no meta.json)")
            continue
        if d.name == prod or _is_decided(hist.get(d.name, [])):
            continue
        live.append(d)
    newest = live[-1].name if live else None
    for d in live:
        if d.name != newest and not (d / "eval.json").exists():
            abandon(d.name, "superseded")
    live = [d for d in live if d.name not in {a["ver"] for a in out["abandoned"]}]
    for d in live:
        try:
            parent = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("parent")
        except (OSError, ValueError):
            parent = None
        if parent != prod:
            abandon(d.name, f"stale parent ({parent or 'BASE'} is not production {prod or 'BASE'})")
        elif not (d / "eval.json").exists() and attempts.get(d.name, 0) >= 2:
            abandon(d.name, "eval failed twice")
    gone = {a["ver"] for a in out["abandoned"]}
    rest = [d.name for d in live if d.name not in gone]
    out["pending"] = rest[-1] if rest else None
    return out


class Run:
    def __init__(self, log):
        self.log = log
        self.stages: list[dict] = []
        self.failed = False

    def stage(self, name: str, fn):
        """fn() -> (status, reason, detail[, exit]); exceptions = failed. After a failure every later stage is skipped."""
        t0 = _now()
        if self.failed:
            res = {"name": name, "status": "skipped", "exit": None, "reason": "an earlier stage failed", "started": t0, "ended": t0, "detail": None}
            self.stages.append(res)
            return None
        touch_lock()
        try:
            r = fn()
            status, reason, detail = r[:3]
            code = r[3] if len(r) > 3 else (EXIT_FAILED if status == "failed" else EXIT_OK)
        except Exception as e:  # noqa: BLE001 - a crashing stage is recorded, not propagated
            status, reason, detail, code = "failed", f"{type(e).__name__}: {e}", {"traceback": traceback.format_exc()[-2000:]}, EXIT_FAILED
        res = {"name": name, "status": status, "exit": code, "reason": reason, "started": t0, "ended": _now(), "detail": detail}
        self.stages.append(res)
        self.log(f"[nightly] {name}: {status}" + (f" - {reason}" if reason else ""))
        if status == "failed":
            self.failed = True
        return res

    def skip(self, name: str, reason: str) -> None:
        t = _now()
        self.stages.append({"name": name, "status": "skipped", "exit": None, "reason": reason, "started": t, "ended": t, "detail": None})


def run(synthetic: bool = False, epochs: int = 2, judge_budget: int | None = None, power=on_ac_power, log=print) -> dict:
    from . import evaluate, promote, train
    started = _now()
    ac = power()
    rec = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}", "kind": "nightly", "synthetic": synthetic,
           "started": started, "host_power": {True: "ac", False: "battery", None: "unknown"}[ac]}
    R = Run(log)
    ctx: dict = {"ver": None}

    def preflight():
        st = config.load_state()
        sw = abandoned_sweep(st, synthetic, log=log)
        config.save_state(st)
        ctx["ver"] = sw["pending"]
        busy = sw["removed_torn"] or sw["abandoned"]
        return ("ok" if busy else "noop"), None, sw

    R.stage("preflight", preflight)
    if ac is False:
        for n in ("train", "eval", "promote"):
            R.skip(n, "on battery power")
    else:
        def do_train():
            if ctx["ver"]:
                return "skipped", f"resuming pending candidate {ctx['ver']}", None
            meta = train.run(synthetic=synthetic, epochs=epochs, log=log)
            if meta is None:
                return "noop", "nothing new to train on", None
            ctx["ver"] = meta["ver"]
            return "ok", None, {"ver": meta["ver"], "n_new": meta.get("n_new"), "n_replay": meta.get("n_replay")}

        def do_eval():
            ver = ctx["ver"]
            if not ver:
                return "noop", "no candidate", None
            if (config.ADAPTERS / ver / "eval.json").exists():
                return "noop", f"{ver} already evaluated", None
            st = config.load_state()
            st.setdefault("eval_attempts", {})[ver] = st.get("eval_attempts", {}).get(ver, 0) + 1
            config.save_state(st)
            res = evaluate.run(synthetic=synthetic, ver=ver, judge_budget=judge_budget, log=log)
            if res is None or not (config.ADAPTERS / ver / "eval.json").exists():
                return "failed", f"eval of {ver} produced no eval.json", None
            return "ok", res["gate"]["reason"], {"ver": ver, "promote": res["gate"]["promote"]}

        def do_promote():
            ver = ctx["ver"]
            if not ver:
                return "noop", "no candidate", None
            r = promote.run(synthetic=synthetic, ver=ver, log=log)
            code = int(r.get("exit", 0))
            status = "failed" if code else ("ok" if r.get("promoted") else "noop")
            return status, r.get("error") or r.get("reason") or (r.get("gate") or {}).get("reason"), {"ver": ver, "promoted": r.get("promoted")}, code

        R.stage("train", do_train)
        R.stage("eval", do_eval)
        R.stage("promote", do_promote)
    rec["stages"] = R.stages
    rec["status"] = "failed" if R.failed else "ok"
    rec["exit"] = EXIT_FAILED if R.failed else EXIT_OK
    rec["ended"] = _now()
    config.MODELS.mkdir(parents=True, exist_ok=True)
    with open(config.RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    st = config.load_state()
    st["last_nightly"] = {"run_id": rec["run_id"], "status": rec["status"], "ended": rec["ended"]}
    config.save_state(st)
    log(f"nightly {rec['status']} (exit {rec['exit']}): " + ", ".join(f"{s['name']}={s['status']}" for s in R.stages))
    return rec
