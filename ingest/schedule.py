"""SFT admission (R122): which candidates get the (scarce) local-judge calls, and the decoy contrast a row must pass.

Why (measured on the first real run, data/lifelong/runs.jsonl): the judge budget (300-400 local calls, ~9 s each) was spent in time order on
dispatch reports, so (a) the 169 user-turn candidates never got a call, (b) every SFT row came from ONE session although 8 sessions exist, and
(c) dispatches whose outcome row had already been written were never reconsidered once the budget ran out (the old code only looked for an SFT row
when it wrote the outcome row). Also the local qwen judge accepts ~40% of decoys (the answer to the nearest *different* request): an absolute score
>= 0.7 is not evidence that the answer fits THIS request.

What this module does, all deterministic and idempotent:
* ORDER: candidates are interleaved round-robin over sessions (most recently active session first), and inside a session user turns and agent
  reports alternate, newest first. A small budget therefore spreads over sessions and sources instead of draining into the first big session.
* GRADE: pass 1 grades every candidate (cache hits and the $0 structural pre-check are free; the rest consumes the budget in the order above).
* DECOY CONTRAST: pass 2 grades, for each candidate whose real grade >= the threshold, a DECOY answer (the answer of the MiniLM-nearest other
  candidate request whose answer is not a paraphrase: response cosine < 0.85, same construction as llm/calibrate.py) against the SAME request.
  A row is admitted iff  decoy_score <= real_score - DECOY_MARGIN.  A decoy that cannot be built (no encoder / no other candidate) admits on the
  real grade alone and the row says so (decoy_score null). A decoy that could not be graded for lack of budget leaves the row PENDING, not rejected:
  the real grade is cached, so the next run only needs the decoy call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DECOY_MARGIN = 0.3
DECOY_MAX_COS = 0.85


@dataclass
class Cand:
    target: str           # id the judge cache / admission is keyed by (dispatch id or turn id)
    task: str             # what the judge sees as the task
    instruction: str      # the SFT instruction text (vectors are taken from this)
    output: str           # the answer being graded
    session: str
    ts: str
    source: str           # "agent_report" | "user_turn"


@dataclass
class Admission:
    grade: dict | None = None
    decoy_score: float | None = None
    decoy_target: str | None = None
    decoy_needed: bool = False
    accepted: bool | None = None      # True admitted, False rejected, None pending (no budget yet)
    why: str = ""
    extra: dict = field(default_factory=dict)


def order(cands: list[Cand]) -> list[Cand]:
    """Round-robin over sessions (latest-active session first); inside a session alternate user turns and agent reports, newest first."""
    by_s: dict[str, dict[str, list[Cand]]] = {}
    last: dict[str, str] = {}
    for c in cands:
        by_s.setdefault(c.session, {"user_turn": [], "agent_report": []}).setdefault(c.source, []).append(c)
        last[c.session] = max(last.get(c.session, ""), c.ts or "")
    lanes: list[list[Cand]] = []
    for s in sorted(by_s, key=lambda x: (last[x], x), reverse=True):
        t = sorted(by_s[s].get("user_turn", []), key=lambda c: (c.ts or "", c.target), reverse=True)
        a = sorted(by_s[s].get("agent_report", []), key=lambda c: (c.ts or "", c.target), reverse=True)
        merged = []
        for i in range(max(len(t), len(a))):
            if i < len(t):
                merged.append(t[i])
            if i < len(a):
                merged.append(a[i])
        lanes.append(merged)
    out = []
    for i in range(max((len(x) for x in lanes), default=0)):
        for lane in lanes:
            if i < len(lane):
                out.append(lane[i])
    return out


def pick_decoys(cands: list[Cand], vs, max_cos: float = DECOY_MAX_COS) -> dict[str, Cand]:
    """target -> decoy candidate (nearest other request, answer not a paraphrase). Empty when vectors are unavailable."""
    if vs is None or len(cands) < 2:
        return {}
    iv = vs.matrix([c.instruction for c in cands])
    rv = vs.matrix([c.output for c in cands])
    if iv is None or rv is None:
        return {}
    out = {}
    for i, c in enumerate(cands):
        sims = iv @ iv[i]
        for j in np.argsort(-sims):
            j = int(j)
            if j == i or cands[j].output.strip() == c.output.strip() or cands[j].instruction.strip() == c.instruction.strip():
                continue
            if float(rv[j] @ rv[i]) < max_cos:
                out[c.target] = cands[j]
                break
    return out


def admit(a: Admission, min_judge: float, margin: float = DECOY_MARGIN) -> None:
    g = a.grade
    if g is None:
        a.accepted, a.why = None, "no judge budget yet"
    elif g["score"] < min_judge:
        a.accepted, a.why = False, "judge score below threshold"
    elif not a.decoy_needed:
        a.accepted, a.why = True, "no decoy available: admitted on the real grade"
    elif a.decoy_score is None:
        a.accepted, a.why = None, "decoy not graded yet (budget)"
    elif a.decoy_score <= g["score"] - margin:
        a.accepted, a.why = True, "real beats decoy"
    else:
        a.accepted, a.why = False, "decoy not rejected by the judge"


def grade_all(judge, cands: list[Cand], vs, min_judge: float, margin: float = DECOY_MARGIN, max_cos: float = DECOY_MAX_COS) -> dict[str, Admission]:
    ordered = order(cands)
    res: dict[str, Admission] = {c.target: Admission() for c in ordered}
    for c in ordered:                                              # pass 1: real grades
        res[c.target].grade = judge.grade(c.target, c.task, c.output)
    decoys = pick_decoys(cands, vs, max_cos)
    for c in ordered:                                              # pass 2: decoy contrast for the candidates that passed pass 1
        a = res[c.target]
        d = decoys.get(c.target)
        if a.grade is not None and a.grade["score"] >= min_judge and d is not None:
            a.decoy_needed, a.decoy_target = True, d.target
            g = judge.grade("decoy:" + c.target, c.task, d.output)
            a.decoy_score = None if g is None else g["score"]
        admit(a, min_judge, margin)
    return res
