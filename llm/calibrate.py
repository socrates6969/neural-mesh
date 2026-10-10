"""`python -m llm calibrate`: the pairwise judge must earn trust BEFORE its verdicts may open the promotion gate.

(1) Decoy test (>= 40 pairs): for a real SFT row, A = the row's own reference answer (a sanity floor: the real,
    reference-consistent answer), B = the answer Claude gave to the NEAREST DIFFERENT request (MiniLM-384 over the requests;
    the decoy's answer must also be dissimilar from the reference, cos < DECOY_MAX_COS, so it is wrong rather than a paraphrase).
    Pass = the judge prefers the real answer in BOTH orders for >= 90% of the pairs.
(2) Deterministic agreement (bench/battery.json math/physics/code items, whose answers are computed or unit-tested): pairs
    (an answer the harness scored correct, an answer it scored wrong) taken from the recorded bench runs. Pass = the judge
    picks the correct one in BOTH orders for >= 85% of the pairs (ties and order disagreements count as misses).
The structural pre-check is OFF here: we are measuring the judge, not the filter. Result: llm/models/pair_calibration.json.
The gate stays CLOSED unless that file says passed for the current rubric version and judge model.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from . import config, data, pairjudge

CALIB = config.MODELS / "pair_calibration.json"
MIN_DECOY_PAIRS, DECOY_PASS = 40, 0.90
MIN_DET_PAIRS, DET_PASS = 20, 0.85
DECOY_MAX_COS = 0.85
DET_CATS = ("math", "physics", "code")
BENCH = config.REPO / "bench"


def decoy_pairs(rows: list[dict], n: int) -> list[dict]:
    """n rows (stable hash order) with a nearest-but-wrong decoy. Uses MiniLM-384 (llm.vectors)."""
    import numpy as np
    from .vectors import Encoder
    enc = Encoder()
    iv = enc.encode([r["instruction"][:2000] for r in rows])
    rv = enc.encode([r["response"][:2000] for r in rows])
    out = []
    for i in sorted(range(len(rows)), key=lambda k: hashlib.sha256(rows[k]["id"].encode()).hexdigest()):
        sims = iv @ iv[i]
        for j in np.argsort(-sims):
            if j != i and float(rv[j] @ rv[i]) < DECOY_MAX_COS and rows[j]["response"].strip() != rows[i]["response"].strip():
                out.append({"id": rows[i]["id"], "question": rows[i]["instruction"], "reference": rows[i]["response"],
                            "real": rows[i]["response"], "decoy": rows[j]["response"], "decoy_id": rows[j]["id"],
                            "req_cos": round(float(sims[j]), 3), "resp_cos": round(float(rv[j] @ rv[i]), 3)})
                break
        if len(out) >= n:
            break
    return out


def _reference(item: dict) -> str:
    if item["kind"] == "code":
        return f"Reference implementation (the tests are checked against it):\n```python\n{item['ref']}\n```"
    return f"The correct final answer is {item['answer']}."


def det_pairs(max_pairs: int = 30, tags=("main", "v2")) -> list[dict]:
    """(correct, wrong) answer pairs from the recorded bench runs: Claude (cond A) correct vs local (cond C) wrong, same item."""
    items = {i["id"]: i for i in json.loads((BENCH / "battery.json").read_text(encoding="utf-8")) if i["cat"] in DET_CATS}
    best: dict[tuple[str, str], dict] = {}
    for t in tags:
        p = BENCH / "results" / t / "raw.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                r = json.loads(line)
                if r["id"] in items and r["cond"] in ("A", "C") and r["text"].strip():
                    best.setdefault((r["id"], r["cond"]), r)
    pairs = []
    for iid, it in items.items():
        a, c = best.get((iid, "A")), best.get((iid, "C"))
        if a and c and a["correct"] and not c["correct"]:
            pairs.append({"id": iid, "cat": it["cat"], "question": it["prompt"], "reference": _reference(it), "real": a["text"], "decoy": c["text"]})
    by_cat: dict[str, list[dict]] = {}
    for p in pairs:
        by_cat.setdefault(p["cat"], []).append(p)
    out, k = [], 0                       # round-robin over categories so one category cannot dominate
    while len(out) < max_pairs and any(by_cat.values()):
        cat = sorted(by_cat)[k % len(by_cat)]
        if by_cat[cat]:
            out.append(by_cat[cat].pop(0))
        k += 1
    return out


def run_pairs(judge: pairjudge.PairwiseJudge, pairs: list[dict], workers: int = 4) -> dict:
    """real vs decoy, both orders. A hit = the real answer wins in both orders."""
    res = judge.compare_many([(p["question"], p["reference"], p["real"], p["decoy"]) for p in pairs], workers)
    judged = [(p, r) for p, r in zip(pairs, res) if r["winner"] != "unjudged"]
    hits = sum(1 for _, r in judged if r["winner"] == "x")
    one_order = [o["who"] == "x" for _, r in judged for o in r["orders"] if o["who"] is not None]
    s = pairjudge.summarize([r for _, r in judged])
    return {"n_pairs": len(pairs), "n_judged": len(judged), "hits": hits, "rate": hits / len(judged) if judged else 0.0,
            "single_order_accuracy": sum(one_order) / len(one_order) if one_order else None, "position_bias": s["position_bias"],
            "wrong_wins": s["y_wins"], "ties_or_disagree": s["ties"], "by_how": s["by_how"],
            "detail": [{"id": p["id"], "winner": r["winner"], "how": r["how"]} for p, r in judged]}


def verdict(decoy: dict, det: dict) -> tuple[bool, str]:
    if decoy["n_judged"] < MIN_DECOY_PAIRS:
        return False, f"only {decoy['n_judged']} decoy pairs judged (< {MIN_DECOY_PAIRS})"
    if decoy["rate"] < DECOY_PASS:
        return False, f"decoy test {decoy['rate']:.3f} < {DECOY_PASS}"
    if det["n_judged"] < MIN_DET_PAIRS:
        return False, f"only {det['n_judged']} deterministic pairs judged (< {MIN_DET_PAIRS})"
    if det["rate"] < DET_PASS:
        return False, f"deterministic agreement {det['rate']:.3f} < {DET_PASS}"
    return True, f"decoy {decoy['rate']:.3f} >= {DECOY_PASS} and deterministic agreement {det['rate']:.3f} >= {DET_PASS}"


def load(path: Path = CALIB) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def trusted(model: str = pairjudge.CLAUDE_MODEL, path: Path = CALIB) -> tuple[bool, str]:
    """may the pairwise judge open the gate? Only with a passing calibration for THIS rubric version and judge model."""
    c = load(path)
    if not c:
        return False, "pairwise judge not calibrated (run `python -m llm calibrate`)"
    if c.get("rubric") != pairjudge.RUBRIC_VERSION or c.get("model") != model:
        return False, f"calibration is for another rubric/model ({c.get('rubric')}/{c.get('model')}); recalibrate"
    return bool(c.get("passed")), "calibrated: " + c.get("reason", "") if c.get("passed") else "calibration FAILED: " + c.get("reason", "")


def run(budget: int = 200, n_decoy: int = 44, max_det: int = 30, workers: int = 4, log=print) -> dict:
    rows, _ = data.load_rows(False)
    ledger = pairjudge.Ledger(kind="calibrate", cap=budget)
    judge = pairjudge.PairwiseJudge(ledger, precheck=False)
    dp, tp = decoy_pairs(rows, n_decoy), det_pairs(max_det)
    log(f"calibrating {judge.model} ({pairjudge.RUBRIC_VERSION}): {len(dp)} decoy pairs, {len(tp)} deterministic pairs, up to {2 * (len(dp) + len(tp))} calls, budget {ledger.remaining()} left today")
    decoy = run_pairs(judge, dp, workers)
    log(f"decoy test: real preferred in both orders {decoy['hits']}/{decoy['n_judged']} = {decoy['rate']:.3f}; position bias {decoy['position_bias']}")
    det = run_pairs(judge, tp, workers)
    log(f"deterministic agreement: {det['hits']}/{det['n_judged']} = {det['rate']:.3f}; position bias {det['position_bias']}")
    ok, why = verdict(decoy, det)
    res = {"passed": ok, "reason": why, "rubric": pairjudge.RUBRIC_VERSION, "model": judge.model, "resolved_model": judge.resolved_model,
           "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "thresholds": {"decoy": DECOY_PASS, "det": DET_PASS, "min_decoy": MIN_DECOY_PAIRS, "min_det": MIN_DET_PAIRS},
           "decoy": decoy, "deterministic": det, "judge_stats": judge.stats, "usage": judge.usage, "ledger": ledger.snapshot(),
           "note": "decoy 'real' answer = the row's own reference (sanity floor); deterministic pairs = recorded bench answers (Claude correct vs local wrong)"}
    config.MODELS.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"calibration {'PASSED' if ok else 'FAILED'}: {why}; fresh calls {judge.stats['fresh_calls']}, tokens {judge.usage}")
    return res
