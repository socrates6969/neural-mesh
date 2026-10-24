"""Best-of-N with verified-majority selection, for training and eval.

N independent attempts (different agents and/or plain resampling), each run headless through train.runner.run.
Selection never looks at the gold answer: attempts are grouped by their final answer and the largest group wins
(self-consistency). An optional `verify(task, text) -> bool` that uses only public information (self-tests, a
sandbox recomputation, a units check) restricts the vote to verified attempts. The chosen attempt is then graded
with train.check.grade, and pass@n (any attempt correct) is reported separately as the oracle ceiling.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import check

KEYED = ("int", "numeric", "mcq")


def answer_key(task: dict, text: str):
    """Comparable form of an attempt's final answer, or None (no answer, or a code task with no answer to vote on)."""
    if task["kind"] not in KEYED:
        return None
    a = check.final_answer(text)
    if not a:
        return None
    if task["kind"] == "mcq":
        return check._letters(a) or None
    v = check._num(a)
    if v is None:
        return None
    if task["kind"] == "int" and abs(v - round(v)) < 1e-9:
        return int(round(v))
    return v


def _same_key(kind, a, b):
    if kind == "numeric":
        return abs(a - b) <= max(0.011, 0.01 * max(abs(a), abs(b)))   # same tolerance as check.check_numeric
    return a == b


def _groups(kind, keyed):
    """keyed: [(index, key)] in attempt order -> [[index, ...], ...]; numeric answers merge within tolerance."""
    groups, reps = [], []
    for i, k in keyed:
        for g, r in zip(groups, reps):
            if _same_key(kind, r, k):
                g.append(i)
                break
        else:
            groups.append([i])
            reps.append(k)
    return groups, reps


def best_of_n(agent, task: dict, n: int = 3, workdir=None, *, pack: str | None = None, verify=None, run_fn=None,
              grade_fn=None, grade_all: bool = True, parallel: bool = True, **run_kw) -> dict:
    """Run `n` attempts of `task` and pick one. `agent` is a runner agent dict or a list of them (cycled).

    Returns a record with the chosen attempt, its grade (`ok`), `agreement` (share of valid attempts in the winning
    group, used as confidence), `pass_at_n`, total cost and per-attempt summaries. Attempts that errored (quota,
    timeout, unparsed output) are left out of the vote, never counted as wrong answers.
    """
    if run_fn is None:
        from ..runner import run as run_fn
    grade_fn = grade_fn or check.grade
    agents = agent if isinstance(agent, (list, tuple)) else [agent]
    workdir = Path(workdir or ".")
    t0 = time.time()

    def attempt(i):
        return run_fn(agents[i % len(agents)], task, pack, workdir / ("try%d" % i), **run_kw)

    if parallel and n > 1:
        with ThreadPoolExecutor(max_workers=n) as ex:
            recs = list(ex.map(attempt, range(n)))
    else:
        recs = [attempt(i) for i in range(n)]

    valid = [i for i, r in enumerate(recs) if not r.get("error") and (r.get("text") or "").strip()]
    verified = {i for i in valid if verify and verify(task, recs[i]["text"])}
    pool = [i for i in valid if i in verified] if verified else valid
    out = {"task": task.get("id"), "field": task.get("field"), "n": n, "valid": len(valid),
           "verified": len(verified) if verify else None, "chosen": None, "answer": None, "agreement": 0.0,
           "tie": False, "method": None, "ok": False, "pass_at_n": None, "text": "",
           "cost": round(sum(float(r.get("cost") or 0) for r in recs), 4), "attempts": []}

    if pool:
        keyed = [(i, answer_key(task, recs[i]["text"])) for i in pool]
        keyed = [(i, k) for i, k in keyed if k is not None]
        if keyed:
            groups, reps = _groups(task["kind"], keyed)
            top = max(len(g) for g in groups)
            best = min((g for g in groups if len(g) == top), key=lambda g: g[0])
            out.update(chosen=best[0], answer=reps[groups.index(best)], tie=sum(len(g) == top for g in groups) > 1,
                       agreement=round(len(best) / len(valid), 3),
                       method="verified-majority" if verified else "majority")
        else:
            # No answer to vote on (code tasks): first verified attempt, or first valid one.
            out.update(chosen=pool[0], agreement=round(len(verified) / len(valid), 3) if verify else 0.0,
                       method="verified-first" if verified else "first")
        out["text"] = recs[out["chosen"]]["text"]

    grades = {}
    if out["chosen"] is not None:
        targets = valid if grade_all else [out["chosen"]]
        grades = {i: bool(grade_fn(task, recs[i]["text"])) for i in targets}
        out["ok"] = grades[out["chosen"]]
        if grade_all:
            out["pass_at_n"] = any(grades.values())
    for i, r in enumerate(recs):
        out["attempts"].append({"agent": r.get("agent"), "error": r.get("error"), "cost": r.get("cost"),
                                "secs": r.get("secs"), "key": answer_key(task, r.get("text") or "") if i in valid else None,
                                "verified": i in verified, "ok": grades.get(i)})
    out["secs"] = round(time.time() - t0, 1)
    return out
