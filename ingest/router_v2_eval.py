"""Offline evaluation of router v2 (hive router-v2; pre-registration in C:/Users/mariu/ruv/hive/findings/router-v2.md).

Units: dispatch rows (one decision each) + benchmark rows grouped by identical task_text (one decision per task, success set = every
agent with a grader-verified success row for that text). Continuations excluded. Order (ts, id). Every system runs online over the
whole stream (predict at a decision, then feed the rows with its live feed); candidates = agents seen so far (global); a decision is
scored when >= 2 agents were seen and it is not an env failure.
  PRIMARY   : correct iff pick in the success set (success dispatch rows: {actual}; benchmark: all solvers). Split by decision index:
              fit [0, .64), val [.64, .80), test [.80, 1].
  SECONDARY : E3 definition, dispatch rows only (non-env): correct iff pick == actual; test = last 20% of dispatch rows.
Systems: majority-so-far, key-majority (field, else project), phase rule, current router (neural-mesh.exe, live config), v2.

CLI: python -m ingest.router_v2_eval --outcomes F --exe neural-mesh.exe --split val|test [--grid] [--v2 lam=1,w_p=.5,beta=.05] [--json OUT]
"""
from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import router_v2 as V
from .shadow import FR_ROSTER, mcnemar_exact_p, rule_pick, wilson

FIT, VAL = 0.64, 0.80
GRID = {"lam": (0.25, 1.0), "w_p": (0.0, 0.5), "beta": (0.05, 0.2)}


# ---- events ----------------------------------------------------------------------------------------------------------------------------------

def build_events(rows: list[dict]) -> list[dict]:
    rows = sorted((r for r in rows if V.fed_kind(r)), key=lambda r: (r.get("ts") or "", r.get("id") or ""))
    solvers: dict[str, set] = defaultdict(set)
    for r in rows:
        if r["kind"] == "benchmark" and r.get("outcome") == "success":
            solvers[r.get("task_text") or ""].add(V.agent_of(r))
    n_dec = len([r for r in rows if r["kind"] == "dispatch"]) + len({r.get("task_text") or "" for r in rows if r["kind"] == "benchmark"})
    n_disp = sum(r["kind"] == "dispatch" for r in rows)
    ev, seen_text, di, dd = [], set(), 0, 0
    for r in rows:
        e = {"row": r, "agent": V.agent_of(r), "decision": False, "primary": None, "secondary": None, "split": None, "dsplit": None}
        env = V.is_env_failure(r)
        if r["kind"] == "dispatch":
            e["decision"] = True
            e["split"] = "fit" if di < FIT * n_dec else ("val" if di < VAL * n_dec else "test")
            e["dsplit"] = "train" if dd < VAL * n_disp else "test"
            di, dd = di + 1, dd + 1
            if not env:
                e["secondary"] = {e["agent"]}
                if r.get("outcome") == "success":
                    e["primary"] = {e["agent"]}
        else:
            t = r.get("task_text") or ""
            if t not in seen_text:
                seen_text.add(t)
                e["decision"] = True
                e["split"] = "fit" if di < FIT * n_dec else ("val" if di < VAL * n_dec else "test")
                di += 1
                e["primary"] = set(solvers[t]) or None
        e["env"] = env
        ev.append(e)
    return ev


def run_stream(events: list[dict], system) -> list:
    """One prediction per event (None when not scored); the system is fed every row after its prediction."""
    seen: list[str] = []
    out = []
    for i, e in enumerate(events):
        pick = None
        if e["decision"] and not e["env"] and len(seen) >= 2:
            pick = system.predict(i, e, list(seen))
        out.append(pick)
        system.feed(i, e)
        if e["agent"] not in seen:
            seen.append(e["agent"])
    system.close()
    return out


# ---- systems ---------------------------------------------------------------------------------------------------------------------------------

class Majority:
    def __init__(self, keyed=False):
        self.keyed, self.g, self.k = keyed, Counter(), defaultdict(Counter)

    def key(self, e):
        r = e["row"]
        return V.field_of(r.get("features")) or (r.get("project") or "default")

    def predict(self, i, e, cands):
        c = self.k[self.key(e)] if self.keyed and self.k[self.key(e)] else self.g
        return c.most_common(1)[0][0]

    def feed(self, i, e):
        self.g[e["agent"]] += 1
        self.k[self.key(e)][e["agent"]] += 1

    def close(self):
        pass


class Rule:
    def predict(self, i, e, cands):
        return rule_pick(e["row"].get("task_text") or "")

    def feed(self, i, e):
        pass

    def close(self):
        pass


class Current:
    """The live Hebbian router through its CLI on a scratch state: label feed, env failures as --failure-class env, late_agent_init 0."""

    def __init__(self, exe: Path):
        from .meshio import clean_features
        self.exe, self.cf = Path(exe), clean_features
        self.work = Path(tempfile.mkdtemp(prefix="v2eval-cur-"))
        self.state = self.work / "state.json"
        self.knobs = ["--late-agent-init", "0"]

    def _run(self, cmd):
        try:
            p = subprocess.run([str(self.exe), *cmd], capture_output=True, text=True, timeout=60)
            return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def predict(self, i, e, cands):
        r = self._run(["route", "--state", str(self.state), "--features", self.cf(e["row"]), "--candidates", ",".join(cands),
                       "--seed", str(7 + i), *self.knobs])
        return (r or {}).get("agent")

    def feed(self, i, e):
        fb = ["--agent", e["agent"], "--reward", "0", "--failure-class", "env"] if e["env"] else ["--label", e["agent"]]
        self._run(["feedback", "--state", str(self.state), "--features", self.cf(e["row"]), *fb, *self.knobs])

    def close(self):
        shutil.rmtree(self.work, ignore_errors=True)


class V2:
    def __init__(self, emb: np.ndarray, cfg: V.Config, greedy=False):
        self.b, self.emb, self.greedy = V.Bandit(cfg), emb, greedy

    def predict(self, i, e, cands):
        r = e["row"]
        return self.b.route(self.emb[i], r.get("project") or "default", V.field_of(r.get("features")), cands, greedy=self.greedy)

    def feed(self, i, e):
        V.apply_outcome(self.b, e["row"], self.emb[i])

    def close(self):
        pass


# ---- scoring ---------------------------------------------------------------------------------------------------------------------------------

def select(events, preds_any, which: str, split: str) -> list[int]:
    if which == "primary":
        return [i for i, e in enumerate(events) if e["primary"] and e["split"] == split and preds_any[i] is not None]
    return [i for i, e in enumerate(events) if e["secondary"] and e["dsplit"] == split and preds_any[i] is not None]


def score(events, idx, pred, which) -> dict:
    ok = [pred[i] in events[i][which] for i in idx]
    k, n = sum(ok), len(idx)
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    out = {"n": n, "k": k, "acc": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}
    if which == "secondary" and n:
        per = defaultdict(lambda: [0, 0])
        for i, c in zip(idx, ok):
            per[events[i]["agent"]][0] += c
            per[events[i]["agent"]][1] += 1
        out["balanced_acc"] = round(sum(a / b for a, b in per.values()) / len(per), 4)
    return out


def paired(events, idx, pa, pb, which) -> dict:
    a = [pa[i] in events[i][which] for i in idx]
    b = [pb[i] in events[i][which] for i in idx]
    x = sum(1 for p, q in zip(a, b) if p and not q)
    y = sum(1 for p, q in zip(a, b) if q and not p)
    return {"a_only": x, "b_only": y, "mcnemar_exact_p": float(f"{mcnemar_exact_p(x, y):.3g}")}


def embeddings(events, cache: Path | None) -> np.ndarray:
    texts = [V.task_text(e["row"].get("task_text"), e["row"].get("features")) for e in events]
    key = hashlib.sha256("\n\x00".join(texts).encode()).hexdigest()[:16]
    if cache:
        f = cache / f"emb-{key}.npy"
        if f.exists():
            return np.load(f)
    E = V.embed(texts)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"emb-{key}.npy", E)
    return E


def parse_cfg(s: str | None) -> V.Config:
    c = V.Config()
    for kv in [x for x in (s or "").split(",") if x]:
        k, v = kv.split("=")
        c = replace(c, **{k: (int(float(v)) if k == "cap" else float(v))})
    return c


def report(events, preds: dict[str, list], split: str, ref: str | None) -> dict:
    any_pred = preds[next(iter(preds))]
    res = {}
    for which, sp in (("primary", split), ("secondary", "test" if split == "test" else "train")):
        if which == "secondary" and split != "test":
            continue                                     # the secondary split has no validation part; it is only read at test time
        idx = select(events, any_pred, which, sp)
        sec = {"split": sp, "systems": {n: score(events, idx, p, which) for n, p in preds.items()}}
        fr = [i for i in idx if events[i]["agent"] in FR_ROSTER and events[i]["row"]["kind"] == "dispatch"]
        if which == "secondary" and fr:                  # E3 gate subset: freight-recovery fr-* dispatches, where the phase rule can be right
            sec["fr_in_domain"] = {"systems": {n: score(events, fr, p, which) for n, p in preds.items()},
                                   "paired_vs": {f"{n} vs {b}": paired(events, fr, preds[n], preds[b], which)
                                                 for n in preds if n.startswith("v2") for b in ("rule", "current") if b in preds}}
        if ref:
            sec["paired_vs"] = {f"{n} vs {b}": paired(events, idx, preds[n], preds[b], which)
                                for n in preds if n.startswith("v2") for b in preds if not b.startswith("v2")}
        res[which] = sec
    return res


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m ingest.router_v2_eval")
    p.add_argument("--outcomes", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--grid", action="store_true", help="validation only: run the pre-registered v2 grid")
    p.add_argument("--v2", default="", help="frozen v2 config, e.g. lam=1,w_p=0.5,beta=0.05")
    p.add_argument("--no-current", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cache", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    if a.grid and a.split == "test":
        p.error("the grid is chosen on validation only")
    rows = [json.loads(l) for l in a.outcomes.read_text(encoding="utf-8").splitlines() if l.strip()]
    events = build_events(rows)
    emb = embeddings(events, a.cache)
    systems = {"majority": lambda: Majority(), "key_majority": lambda: Majority(keyed=True), "rule": lambda: Rule()}
    if not a.no_current:
        systems["current"] = lambda: Current(a.exe)
    if a.grid:
        for lam, wp, beta in itertools.product(GRID["lam"], GRID["w_p"], GRID["beta"]):
            c = replace(V.Config(), lam=lam, w_p=wp, beta=beta)
            systems[f"v2_lam{lam:g}_wp{wp:g}_b{beta:g}"] = (lambda c=c: V2(emb, c))
    else:
        c = parse_cfg(a.v2)
        systems["v2"] = lambda: V2(emb, c)
        systems["v2_greedy_ablation"] = lambda: V2(emb, c, greedy=True)
    with V.blas_threads(), ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {n: ex.submit(run_stream, events, f()) for n, f in systems.items()}
        preds = {n: f.result() for n, f in futs.items()}
    res = {"split": a.split, "n_events": len(events), "n_decisions": sum(e["decision"] for e in events),
           "v2_config": None if a.grid else asdict(parse_cfg(a.v2)), **report(events, preds, a.split, "current" if not a.no_current else None)}
    if a.json:
        res["preds"] = {n: [p if (e["primary"] or e["secondary"]) else None for p, e in zip(v, events)] for n, v in preds.items()}
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
        del res["preds"]
    for which in ("primary", "secondary"):
        if which not in res:
            continue
        print(f"== {which} ({res[which]['split']})")
        for n, s in res[which]["systems"].items():
            ba = f" ba={s['balanced_acc']}" if "balanced_acc" in s else ""
            print(f"  {n:28s} acc={s['acc']} ({s['k']}/{s['n']}) ci95={s['ci95']}{ba}")
        for k, v in (res[which].get("paired_vs") or {}).items():
            print(f"  {k:40s} {v}")
        if "fr_in_domain" in res[which]:
            print("  -- fr-* in-domain dispatches")
            for n, s in res[which]["fr_in_domain"]["systems"].items():
                print(f"  {n:28s} acc={s['acc']} ({s['k']}/{s['n']}) ci95={s['ci95']}")
            for k, v in res[which]["fr_in_domain"]["paired_vs"].items():
                print(f"  {k:40s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())