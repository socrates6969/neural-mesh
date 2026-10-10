"""Build bench/data/ruflo_dispatch_v2.jsonl: freight-recovery dispatches + SendMessage continuations (each is a "which fr-* role does this unit of work go to"
decision, same convention as the router-eval dataset behind bench/data/ruflo_dispatch.jsonl) with the v2 STRUCTURAL labels (ingest/labeler.py) instead of the
v1 text-cue labels. Rows are scrubbed already (ingest state). Only chains whose role is one of the eight fr-* roles are kept (statsclaw:X -> fr-X; a
general-purpose agent told 'You are acting as the StatsClaw **X**' -> fr-X; the distiller has no fr-* equivalent and is dropped).
Run from the repo root:  python bench/build_ruflo_dataset.py   (needs data/lifelong/state; the output is committed, the state is not)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ingest.config import Config  # noqa: E402
from ingest.labeler import build_items, label_items, role_of  # noqa: E402
from ingest.shadow import rule_pick  # noqa: E402

ROLE_TO_FR = {"planner": "fr-planner", "builder": "fr-builder", "tester": "fr-tester", "reviewer": "fr-reviewer", "scriber": "fr-scriber", "shipper": "fr-shipper"}
FR_AGENTS = {"fr-planner", "fr-builder", "fr-tester", "fr-reviewer", "fr-scriber", "fr-shipper", "fr-ci-diagnoser", "fr-security-reviewer"}
OUT_NAME = {"success": "passed", "defect": "defects", "fix-round": "fix-round", "re-dispatch": "re-dispatch", "blocked": "blocked"}


def main():
    cfg = Config()
    items = build_items(cfg)
    labels = label_items(items)
    head = {}
    for it in sorted(items, key=lambda x: (x.ts or "", x.id)):
        head.setdefault(it.chain, it)
    rows = []
    for it in sorted(items, key=lambda x: (x.ts or "", x.id)):
        if it.project != "freight-recovery":
            continue
        h = head[it.chain]
        raw = (h.agent or "").split(":")[-1]
        if raw.startswith("fr-") and raw in FR_AGENTS:
            fr = raw
        else:
            role = role_of(h.agent, h.prompt)
            fr = ROLE_TO_FR.get(role)
        if not fr:
            continue
        msg = " ".join((it.prompt or "")[:240].split())
        rows.append({"ts": it.ts, "task_text": (it.description + " :: " + msg).strip(" :"), "actual_agent": fr, "outcome": OUT_NAME[labels[it.id][0]],
                     "phase_rule": rule_pick(it.description + " :: " + msg), "kind": it.kind, "chain": it.chain[-6:], "label_signals": labels[it.id][1]})
    out = ROOT / "bench" / "data" / "ruflo_dispatch_v2.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    from collections import Counter
    print(len(rows), Counter(r["actual_agent"] for r in rows), Counter(r["outcome"] for r in rows))


if __name__ == "__main__":
    main()
