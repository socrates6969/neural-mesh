"""mesh-ops alerts: diff the current state against the last snapshot and raise Windows toasts.

    py -3.12 ops/alerts.py [--out DIR] [--nm-root DIR] [--ac-root DIR] [--dry-run] [--test-toast]

Events: pack_promoted, quota_stop, stage_done, nightly_promoted, nightly_failed, router_stale, training_error,
workspace_verified. The first run only records a baseline (no toasts for history). Each event type is toasted at
most once per hour; further events of that type inside the hour are logged as "suppressed". router_stale fires
when the router turns stale and then reminds at most once per 24 h while it stays stale.
Every event is appended to <out>/alerts.log. Snapshot: <out>/alert_state.json.
Delivery: BurntToast if installed, else a WinRT toast via [Windows.UI.Notifications], else msg.exe.
Set OPS_NOTIFY=none (or --dry-run) to log without showing anything.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import sources  # noqa: E402

RATE_LIMIT_S = 3600
STALE_REMIND_S = 24 * 3600

PS_TOAST = r"""
$ErrorActionPreference = 'Stop'
$t = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__T__'))
$m = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__M__'))
if (Get-Module -ListAvailable -Name BurntToast) {
  Import-Module BurntToast; New-BurntToastNotification -Text $t, $m; 'burnttoast'; exit 0 }
try {
  [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
  [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null
  $x = New-Object Windows.Data.Xml.Dom.XmlDocument
  $e = [Security.SecurityElement]
  $x.LoadXml('<toast><visual><binding template="ToastGeneric"><text>' + $e::Escape($t) + '</text><text>' + $e::Escape($m) + '</text></binding></visual></toast>')
  $app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
  [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($x))
  'winrt'; exit 0
} catch { }
$msg = Join-Path $env:WINDIR 'System32\msg.exe'
if (Test-Path $msg) { & $msg $env:USERNAME /TIME:600 ($t + ': ' + $m); 'msg'; exit 0 }
'none'; exit 1
"""


def notify(title: str, msg: str) -> str:
    """Show one toast; returns the delivery method used ('burnttoast' | 'winrt' | 'msg' | 'none' | 'disabled')."""
    if os.environ.get("OPS_NOTIFY", "").lower() == "none":
        return "disabled"
    b64 = lambda s: base64.b64encode(s.encode("utf-8")).decode("ascii")  # noqa: E731
    script = PS_TOAST.replace("__T__", b64(title[:120])).replace("__M__", b64(msg[:400]))
    enc = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                            "-EncodedCommand", enc], capture_output=True, text=True, timeout=60,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as ex:
        return "error: %s" % ex
    out = (r.stdout or "").strip().splitlines()
    return out[-1] if out else "error: exit %s %s" % (r.returncode, (r.stderr or "").strip()[:200])


# ---------------------------------------------------------------- snapshot and diff

def snapshot(st: dict) -> dict:
    tr, me, ni = st["training"], st["mesh"], st["nightly"]
    return {
        "packs": {"%s/%s" % (b["name"], k): p["status"] for b in tr["batches"] for k, p in b["packs"].items()},
        "result_errors": {b["name"]: b["result_errors"] for b in tr["batches"]},
        "logs": {p: {"size": lg["size"], "quota": len(lg["quota"]), "stage": len(lg["stage"]),
                     "traceback": len(lg["traceback"])} for p, lg in tr["logs"].items()},
        "promotions": len(me["promotions"]),
        "nightly_seen": [r["run_id"] for r in ni["runs"]],
        "router_stale": bool(me["router"]["stale"]),
        "workspace_max_verified": me["workspace"]["max_verified_id"],
    }


def diff(st: dict, prev: dict) -> list[dict]:
    """Events between the previous snapshot and the current state (prev must be a snapshot, not None)."""
    ev = []
    tr, me, ni = st["training"], st["mesh"], st["nightly"]

    def add(kind, title, msg):
        ev.append({"type": kind, "title": title, "msg": msg})

    for b in tr["batches"]:
        for k, p in b["packs"].items():
            key = "%s/%s" % (b["name"], k)
            if p["status"] == "promoted" and prev.get("packs", {}).get(key) != "promoted":
                add("pack_promoted", "Pack promoted", "%s (%s) v%s, pooled p=%s" % (
                    k, b["name"], p["version"], (p["pooled"] or {}).get("p")))
        before = prev.get("result_errors", {}).get(b["name"])
        if before is not None and b["result_errors"] > before:
            add("training_error", "Training errors", "%s: %d new result rows with an error" % (
                b["name"], b["result_errors"] - before))
    for p in me["promotions"][prev.get("promotions", 0):]:
        if p.get("result") == "published":
            add("pack_promoted", "Pack published to live mesh", "%s / %s v%s" % (p.get("agent"), p.get("field"), p.get("version")))
    for path, lg in tr["logs"].items():
        old = prev.get("logs", {}).get(path)
        if old is None or lg["size"] < old.get("size", 0):
            old = {"quota": 0, "stage": 0, "traceback": 0}  # new or truncated log: everything in it is new
        name = Path(path).name if Path(path).name != "train.log" else Path(path).parent.name + "/train.log"
        for line in lg["quota"][old["quota"]:]:
            add("quota_stop", "Training quota stop", "%s: %s" % (name, line[:300]))
        for line in lg["stage"][old["stage"]:]:
            add("stage_done", "Training stage done", "%s: %s" % (name, line[:300]))
        n_tb = len(lg["traceback"]) - old["traceback"]
        if n_tb > 0:
            add("training_error", "Training error", "%s: %d new Traceback(s)" % (name, n_tb))
    seen = set(prev.get("nightly_seen", []))
    for r in ni["runs"][::-1]:
        if r["run_id"] in seen:
            continue
        if r["exit"] not in (0, None) or r["status"] not in ("ok", None):
            add("nightly_failed", "Nightly LLM failed", "%s status %s exit %s" % (r["run_id"], r["status"], r["exit"]))
        elif r["promoted"]:
            add("nightly_promoted", "Nightly LLM promoted", "%s -> production. %s" % (r["ver"], (r["gate"] or "")[:200]))
    rt = me["router"]
    if rt["stale"] and not prev.get("router_stale"):
        add("router_stale", "Router not updated", "router_state.json last written %s (%.0f h ago)" % (
            (rt["state"] or {}).get("mtime_iso", "never"), rt["age_h"] or 0))
    ws = me["workspace"]
    if ws["max_verified_id"] > prev.get("workspace_max_verified", 0):
        newv = [r for r in ws["recent"] if r["verified"] and r["id"] > prev.get("workspace_max_verified", 0)]
        what = ", ".join("%s by %s" % (r["task"], r["agent"]) for r in newv[:3]) or "id %d" % ws["max_verified_id"]
        add("workspace_verified", "New verified workspace result", what)
    return ev


def run(over: dict | None = None, dry_run: bool = False, now: float | None = None, st: dict | None = None) -> list[dict]:
    """One alert pass. Returns the events with their delivery outcome ('sent:<method>', 'suppressed', ...)."""
    now = time.time() if now is None else now
    st = st or sources.collect(over)
    out = sources.roots(over)["out"]
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "alert_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = None
    cur = snapshot(st)
    if not state or "snapshot" not in state:
        _log(out, now, [{"type": "baseline", "title": "baseline", "msg": "first run, snapshot recorded",
                         "outcome": "logged"}])
        _save(state_path, {"snapshot": cur, "last_sent": {}, "router_stale_reminded": None})
        return []
    events = diff(st, state["snapshot"])
    last_sent = state.get("last_sent", {})
    reminded = state.get("router_stale_reminded")
    if cur["router_stale"] and state["snapshot"].get("router_stale") and reminded and now - reminded >= STALE_REMIND_S:
        rt = st["mesh"]["router"]
        events.append({"type": "router_stale", "title": "Router still not updated",
                       "msg": "router_state.json %.0f h old" % (rt["age_h"] or 0)})
    for evt in events:
        if evt["type"] == "router_stale":
            reminded = now
        if now - last_sent.get(evt["type"], 0) < RATE_LIMIT_S:
            evt["outcome"] = "suppressed"
            continue
        method = "dry-run" if dry_run else notify(evt["title"], evt["msg"])
        evt["outcome"] = "sent:" + method
        last_sent[evt["type"]] = now
    if not cur["router_stale"]:
        reminded = None
    _log(out, now, events)
    _save(state_path, {"snapshot": cur, "last_sent": last_sent, "router_stale_reminded": reminded})
    return events


def _save(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def _log(out: Path, now: float, events: list[dict]) -> None:
    if not events:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
    with open(out / "alerts.log", "a", encoding="utf-8") as f:
        for evt in events:
            f.write("%s\t%s\t%s\t%s: %s\n" % (ts, evt["type"], evt.get("outcome", ""), evt["title"],
                                              evt["msg"].replace("\n", " ")))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out")
    p.add_argument("--nm-root")
    p.add_argument("--ac-root")
    p.add_argument("--dry-run", action="store_true", help="log events, show no toast")
    p.add_argument("--test-toast", action="store_true", help="show one test toast and print the delivery method")
    a = p.parse_args(argv)
    if a.test_toast:
        print("delivery:", notify("neural-mesh ops", "test notification from ops/alerts.py"))
        return 0
    over = {k: v for k, v in (("out", a.out), ("nm", a.nm_root), ("ac", a.ac_root)) if v}
    events = run(over, dry_run=a.dry_run)
    for evt in events:
        print("%s [%s] %s: %s" % (evt["type"], evt["outcome"], evt["title"], evt["msg"]))
    print("alerts: %d event(s)" % len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())