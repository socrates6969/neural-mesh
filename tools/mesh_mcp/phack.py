"""phack_guard: checks a set of statistical tests for multiple testing without correction, optional stopping
(peeking), unregistered outcomes and p-values clustered just under .05. Pure Python; runs on the host.

stats_test appends every call to a per-session JSONL log (record_test); phack_guard reads that log when no
tests are passed in. Verdict: BLOCK if any blocking rule fires, otherwise OK (with warnings).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path

from tools.mesh_mcp.common import as_json, cap

DEFAULT_DIR = r"C:\Users\mariu\ruv\hive\runs\stats-log"


def log_path(session: str | None = None) -> Path:
    s = session or os.environ.get("CLAUDE_SESSION_ID") or "default"
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:80]
    return Path(os.environ.get("MESH_STATS_LOG_DIR", DEFAULT_DIR)) / ("%s.jsonl" % safe)


def record_test(kind: str, data: dict, result: dict, label: str | None = None, look: int | None = None) -> None:
    """Append one stats_test call (never raises: logging must not break the tool)."""
    try:
        rec = {"ts": time.time(), "kind": kind, "label": label, "look": look, "p": result.get("p"),
               "n": result.get("n") or result.get("n_pairs") or
               (result.get("n_a", 0) + result.get("n_b", 0) if "n_a" in result else None)}
        if kind == "multitest":
            rec.update({"p_raw": result.get("p_raw"), "method": result.get("test"), "p": None})
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


def read_log(session: str | None = None) -> list[dict]:
    p = log_path(session)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def holm(ps: list[float], alpha: float) -> list[bool]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    reject, m = [False] * len(ps), len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def check(tests: list[dict], prereg: dict | None = None, alpha: float = 0.05, caliper: float = 0.005) -> dict:
    block, warn, info = [], [], []
    prereg = prereg or {}
    alpha = float(prereg.get("alpha", alpha))
    corrections = [t for t in tests if t.get("kind") == "multitest"]
    tests = [t for t in tests if t.get("kind") != "multitest" and isinstance(t.get("p"), (int, float))]
    if not tests:
        return {"verdict": "OK", "block": [], "warn": ["no tests with p-values to check"], "info": [], "m": 0}
    ps = [float(t["p"]) for t in tests]
    m = len(ps)
    info.append("%d tests, %d with p < %g" % (m, sum(p < alpha for p in ps), alpha))

    # 1) multiple testing without correction. The family is one entry per outcome (its last look; repeated
    # looks at the same outcome are judged by the optional-stopping rule below, not counted as extra tests).
    last = {}
    for i, t in enumerate(tests):
        last[t.get("label") or ("#%d" % i)] = i
    fam = sorted(last.values())
    fam_tests, fam_ps = [tests[i] for i in fam], [ps[i] for i in fam]
    corrected = set()
    for c in corrections:
        for p in c.get("p_raw") or []:
            corrected.add(round(float(p), 12))
    declared = str(prereg.get("correction", "")).lower() in ("holm", "bh", "fdr_bh", "bonferroni")
    k = len(fam_ps)
    sig = [i for i, p in enumerate(fam_ps) if p < alpha]
    uncorrected = [i for i in sig if round(fam_ps[i], 12) not in corrected]
    if k > 1 and uncorrected and not declared:
        survive = holm(fam_ps, alpha)
        lost = [i for i in uncorrected if not survive[i]]
        if lost:
            block.append("multiple testing without correction: %d outcomes tested, %d raw-significant; %d of those "
                         "(%s) do not survive Holm at %g. Run stats_test kind=multitest on all %d p-values and report "
                         "adjusted values." % (k, len(sig), len(lost), ", ".join(_name(fam_tests[i]) for i in lost),
                                               alpha, k))
        else:
            warn.append("%d outcomes tested and no correction recorded; all raw-significant results also survive "
                        "Holm at %g, but report the correction" % (k, alpha))

    # 2) optional stopping / peeking: the same outcome tested repeatedly
    by_label = defaultdict(list)
    for t in tests:
        if t.get("label"):
            by_label[t["label"]].append(t)
    planned = int(prereg.get("looks", 1))
    sequential = bool(prereg.get("sequential") or prereg.get("alpha_spending"))
    for label, ts in by_label.items():
        if len(ts) > 1:
            ns = [t.get("n") for t in ts]
            looks = len(ts)
            stopped_on_sig = float(ts[-1]["p"]) < alpha and all(float(t["p"]) >= alpha for t in ts[:-1])
            msg = "outcome %r tested %d times (n = %s)" % (label, looks, ", ".join(str(n) for n in ns))
            if sequential and looks <= planned:
                info.append(msg + "; within the preregistered sequential plan (%d looks)" % planned)
            elif stopped_on_sig or looks > planned:
                block.append(msg + ("; stopped at the first significant look" if stopped_on_sig else "") +
                             ": optional stopping inflates false positives. Use a preregistered sequential design "
                             "(alpha spending) or report only the planned final look.")
            else:
                warn.append(msg + "; repeated looks at the same outcome")

    # 3) unregistered outcomes
    registered = prereg.get("outcomes")
    if registered is not None:
        reg = {str(o) for o in registered}
        extra = sorted({t.get("label") or "(unlabelled)" for t in tests} - reg)
        if extra:
            block.append("outcomes not in the preregistration: %s (registered: %s). Report them as exploratory or "
                         "drop them." % (", ".join(extra), ", ".join(sorted(reg))))
        missing = sorted(reg - {t.get("label") for t in tests})
        if missing:
            warn.append("registered outcomes with no test: %s (report them, including null results)"
                        % ", ".join(missing))

    # 4) caliper test: p-values just under alpha vs just over
    below = sum(alpha - caliper < p <= alpha for p in ps)
    above = sum(alpha < p <= alpha + caliper for p in ps)
    if below + above >= 5:
        pc = _binom_sf(below, below + above)
        line = "caliper test (+/-%g around %g): %d just below, %d just above, one-sided binomial p=%.3g" % (
            caliper, alpha, below, above, pc)
        (block if pc < 0.05 else info).append(line + (": p-values bunch just under the threshold"
                                                      if pc < 0.05 else ""))
    elif below >= 2:
        warn.append("%d p-values in (%g, %g]; too few around the threshold for a caliper test"
                    % (below, alpha - caliper, alpha))
    return {"verdict": "BLOCK" if block else "OK", "block": block, "warn": warn, "info": info, "m": m}


def _name(t):
    return t.get("label") or t.get("name") or t.get("kind") or "test"


def phack_guard(tests_json=None, prereg_json=None, alpha: float = 0.05, session: str | None = None) -> str:
    if tests_json in (None, "", []):
        tests, src = read_log(session), "session log %s" % log_path(session)
    else:
        tests, src = as_json(tests_json, "tests_json"), "tests_json"
        if not isinstance(tests, list):
            raise ValueError("tests_json must be a list of {label, p, n?, look?, kind?}")
        tests = [dict(t, label=t.get("label") or t.get("outcome") or t.get("name")) for t in tests]
    prereg = as_json(prereg_json, "prereg_json") if prereg_json not in (None, "") else None
    r = check(tests, prereg, alpha)
    lines = ["%s (%s; %d tests checked)" % (r["verdict"], src, r["m"])]
    lines += ["BLOCK: " + x for x in r["block"]] + ["WARN: " + x for x in r["warn"]] + ["info: " + x for x in r["info"]]
    return cap("\n".join(lines))
