"""Wall-clock deadline for the nightly llm round (MESH_NIGHTLY_DEADLINE "HH:MM", local; run_nightly.ps1 sets 07:30).

Budget order inside the round: training keeps its own TRAIN_MINUTES budget; then eval answers the deterministic battery in
full, generates pairwise answers in chunks until JUDGE_RESERVE_MIN + PROMOTE_RESERVE_MIN before the deadline, and starts
judge pairs until PROMOTE_RESERVE_MIN before it (a started pair always finishes). If that cut leaves the win-rate CI
undecided, the eval records "no decision" and production stays (evaluate.decide)."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from . import config


def deadline_ts(spec: str | None = None, now: datetime | None = None) -> float | None:
    """epoch seconds of the next HH:MM after now (a round started at 03:30 with 07:30 ends the same morning; a daytime
    manual run gets tomorrow's 07:30, i.e. no practical limit). None when unset or unparsable."""
    spec = config.NIGHTLY_DEADLINE if spec is None else spec
    try:
        hh, mm = (int(x) for x in spec.strip().split(":"))
        now = now or datetime.now()
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    except (ValueError, AttributeError):
        return None
    if t <= now:
        t += timedelta(days=1)
    return t.timestamp()


def gen_stop(dl: float | None) -> float | None:
    return None if dl is None else dl - (config.JUDGE_RESERVE_MIN + config.PROMOTE_RESERVE_MIN) * 60


def judge_stop(dl: float | None) -> float | None:
    return None if dl is None else dl - config.PROMOTE_RESERVE_MIN * 60


def passed(t: float | None) -> bool:
    return t is not None and time.time() >= t


def fmt(t: float | None) -> str:
    return "none" if t is None else datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M")