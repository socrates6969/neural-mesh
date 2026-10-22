"""Online A/B O2 design A scorer: paired runs (same task with / without brief), metrics read from subagent transcripts.

pairs.json: [{"task": "...", "brief_agent": "<agentId of run with brief>", "nobrief_agent": "<agentId>", "brief_ok": true, "nobrief_ok": true}, ...]
python -m prebrief.online pairs.json   (agentId = the part after "agent-" in ~/.claude/projects/<proj>/<session>/subagents/agent-<id>.jsonl)
"""
import glob
import json
import os
import statistics
import sys

from .corpus import PROJECTS, read_run
from .evaluate import mcnemar_exact, mde_sign


def _metrics(agent_id: str) -> dict:
    hits = glob.glob(os.path.join(PROJECTS, "*", "*", "subagents", f"agent-{agent_id}.jsonl"))
    if not hits:
        raise SystemExit(f"no transcript for agent {agent_id}")
    return read_run(hits[0])["metrics"]


def score(pairs: list) -> dict:
    rows = []
    for p in pairs:
        b, n = _metrics(p["brief_agent"]), _metrics(p["nobrief_agent"])
        rows.append({"b": b, "n": n, "bok": p.get("brief_ok"), "nok": p.get("nobrief_ok")})
    out = {"n_pairs": len(rows)}
    for m in ("fresh_in", "out", "turns", "tools", "wall_s"):
        d = [r["b"][m] - r["n"][m] for r in rows]
        rel = [(r["b"][m] - r["n"][m]) / r["n"][m] for r in rows if r["n"][m]]
        wins, losses = sum(x < 0 for x in d), sum(x > 0 for x in d)
        out[m] = {"median_rel_change": round(statistics.median(rel), 3) if rel else None, "brief_lower": wins, "brief_higher": losses,
                  "sign_p": round(mcnemar_exact(wins, losses), 5), "mde_win_share": mde_sign(wins + losses)}
    b = sum(1 for r in rows if r["bok"] and not r["nok"])
    c = sum(1 for r in rows if r["nok"] and not r["bok"])
    out["success"] = {"brief_only": b, "nobrief_only": c, "mcnemar_p": round(mcnemar_exact(b, c), 5)}
    fi = out["fresh_in"]
    out["pass"] = bool(fi["median_rel_change"] is not None and fi["median_rel_change"] <= -0.15 and fi["sign_p"] < 0.05 and b - c >= 0)
    return out


if __name__ == "__main__":
    print(json.dumps(score(json.load(open(sys.argv[1], encoding="utf-8"))), indent=1))