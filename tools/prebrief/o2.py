"""O2 design A (pre-registered in ruv/hive/findings/prebrief.md section 0): build the paired task set and score it.

build: python -m prebrief.o2 build --exclude-session <queen session id>    -> runs/prebrief/o2/{tasks.json, prompts/*.md, protocol.md}
score: python -m prebrief.o2 score                                         -> runs/prebrief/o2/{pairs.json, result.json}
Dispatch tag (the Agent `description`): "O2-<task id>-brief" or "O2-<task id>-nobrief". The scorer finds runs by that tag in the
subagent .meta.json files, grades the final report against the answer key written at build time, and applies the pass rule.
"""
import glob
import json
import os
import random
import re
import sys
from datetime import datetime, timezone

from .brief import RUNS, load_index, prebrief
from .corpus import PROJECTS, read_run

OUT = os.path.join(RUNS, "o2")
FOOTER = ("\n\nRead-only task: do not edit, create or delete any file, do not run builds or tests. "
          "Finish with one line `ANSWER: <answer>` giving the requested file path(s), function name(s) or value(s).")

# (id, repo hint, task text, answer key = regexes that must ALL match the final report, case-insensitive). Written before any dispatch.
# The task list and answer keys are private (they describe the owner's own repositories), so they are kept
# outside the public repo in <RUNS>/o2/candidates.json: [[id, task text, [answer-key regexes]], ...].
def _load_candidates():
    p = os.path.join(RUNS, "o2", "candidates.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [tuple(x) for x in json.load(f)]


CANDIDATES = _load_candidates()


def build(exclude_session: str | None, n: int = 24, seed: int = 20261019):
    os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    idx = load_index()
    rng = random.Random(seed)
    tasks, rejected = [], []
    for tid, text, key in CANDIDATES:
        b = prebrief(text, "Explore", session=exclude_session, idx=idx, before=now)
        md = b["markdown"]
        leaked = [k for k in key if re.search(k, md, re.I)]
        if len(leaked) == len(key):  # the brief alone answers it -> not allowed
            rejected.append({"id": tid, "reason": "brief contains the full answer key", "leaked": leaked})
            continue
        if len(tasks) == n:
            break
        first = rng.choice(["brief", "nobrief"])
        tasks.append({"id": tid, "task": text, "answer_key": key, "partial_key_in_brief": leaked, "first_arm": first,
                      "brief_tokens": b["tokens"], "brief_doc_ids": b["doc_ids"],
                      "tags": {"brief": f"O2-{tid}-brief", "nobrief": f"O2-{tid}-nobrief"}})
        with open(os.path.join(OUT, "prompts", f"{tid}-nobrief.md"), "w", encoding="utf-8") as f:
            f.write(text + FOOTER)
        with open(os.path.join(OUT, "prompts", f"{tid}-brief.md"), "w", encoding="utf-8") as f:
            f.write(md + "\n\n---\n\n" + text + FOOTER)
    meta = {"built_at": now, "brief_before": now, "excluded_session": exclude_session, "seed": seed, "n": len(tasks),
            "rejected": rejected, "pass_rule": "median fresh-token change <= -15%, exact sign test p < .05, no success loss (brief_only - nobrief_only >= 0)",
            "agent_type": "Explore", "tasks": tasks}
    with open(os.path.join(OUT, "tasks.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta


def _find(tag: str):
    for mp in glob.glob(os.path.join(PROJECTS, "*", "*", "subagents", "agent-*.meta.json")):
        try:
            m = json.load(open(mp, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if m.get("description") == tag:
            r = read_run(mp[:-10] + ".jsonl")
            if r:
                return r, m
    return None, None


def score():
    from .online import score as score_pairs
    meta = json.load(open(os.path.join(OUT, "tasks.json"), encoding="utf-8"))
    pairs, missing = [], []
    for t in meta["tasks"]:
        rb, mb = _find(t["tags"]["brief"])
        rn, mn = _find(t["tags"]["nobrief"])
        if not (rb and rn):
            missing.append(t["id"])
            continue
        ok = lambda r: all(re.search(k, r["report"], re.I) for k in t["answer_key"])
        pairs.append({"task": t["id"], "brief_agent": rb["id"].split(":")[-1], "nobrief_agent": rn["id"].split(":")[-1],
                      "brief_ok": ok(rb), "nobrief_ok": ok(rn), "brief_type": mb.get("agentType"), "nobrief_type": mn.get("agentType"),
                      "brief_model": mb.get("model"), "nobrief_model": mn.get("model")})
    with open(os.path.join(OUT, "pairs.json"), "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=1)
    res = score_pairs(pairs) if pairs else {}
    res["missing"] = missing
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "score"
    if cmd == "build":
        ex = sys.argv[sys.argv.index("--exclude-session") + 1] if "--exclude-session" in sys.argv else None
        m = build(ex)
        print(json.dumps({k: m[k] for k in ("n", "rejected", "brief_before")}, indent=1))
    else:
        print(json.dumps(score(), indent=1))