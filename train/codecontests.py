"""Build a harder coding benchmark from DeepMind CodeContests (Codeforces problems, test+valid splits).

A problem is kept only if a reference Python 3 solution passes all of its stored tests inside the sandbox; when two
or more reference solutions exist, two must pass (this filters out problems that accept many different outputs).
Tests per problem: public, then private, then generated, capped at MAX_TESTS.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
from pathlib import Path

from . import check, data

MAX_TESTS = 25
PY3 = 3


def _tests(r):
    ins, outs = [], []
    for k in ("public_tests", "private_tests", "generated_tests"):
        t = r.get(k) or {}
        for i, o in zip(t.get("input", []), t.get("output", [])):
            if len(i) + len(o) < 200_000:
                ins.append(i); outs.append(o)
    return {"inputs": ins[:MAX_TESTS], "outputs": outs[:MAX_TESTS]}


def candidates():
    out = []
    for split in ("test", "valid"):
        for r in data._rows("deepmind/code_contests", "default", split):
            sols = r.get("solutions") or {}
            py = [s for lang, s in zip(sols.get("language", []), sols.get("solution", [])) if lang == PY3]
            tests = _tests(r)
            rating = int(r.get("cf_rating") or 0)
            if not py or len(tests["inputs"]) < 3 or not rating or r.get("input_file") or r.get("output_file"):
                continue
            tl = (r.get("time_limit") or {}).get("seconds") or 2
            out.append({"id": "cc-" + str(r["cf_contest_id"]) + str(r["cf_index"]), "field": "coding", "kind": "stdio",
                        "source": "deepmind/code_contests " + split, "rating": rating,
                        "prompt": r["description"].strip() + "\n\nTime limit per test: %d s (CPython: allow ~3x)." % tl,
                        "time_limit": float(max(4, 3 * tl)), "tests": tests, "answer": "", "_refs": py[:2]})
    return out


def validate(t):
    ok = [check.check_stdio("", t, code=s) for s in t["_refs"]]
    return all(ok) and len(ok) >= 1


def main(out_path):
    cands = candidates()
    print(len(cands), "candidates with Python refs", flush=True)
    with cf.ThreadPoolExecutor(6) as ex:
        keep = [t for t, ok in zip(cands, ex.map(validate, cands)) if ok]
    for t in keep:
        t.pop("_refs")
    Path(out_path).write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in keep) + "\n", encoding="utf-8")
    rs = sorted(t["rating"] for t in keep)
    print(len(keep), "validated; ratings", rs[:3], "...", rs[-3:], "| >=1600:", sum(r >= 1600 for r in rs),
          ">=1900:", sum(r >= 1900 for r in rs), "1200-1500:", sum(1200 <= r <= 1500 for r in rs))


if __name__ == "__main__":
    main(sys.argv[1])