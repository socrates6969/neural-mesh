"""CLI for the shared method-card memory.

  python -m train.memory build [--llm] [batch ...]         cards from verified runs (+ migrate experience.jsonl)
  python -m train.memory eval [--no-llm]                    offline retrieval eval (writes memory/eval.json)
  python -m train.memory eval-graph                         Graph-RAG re-ranking experiment (result: not shipped)
  python -m train.memory hygiene                            drop ambiguous-id items, tag split/problem_key (idempotent)
  python -m train.memory update                             usefulness counts from training results.jsonl
  python -m train.memory serve                              run the localhost daemon (normally started on demand)
  python -m train.memory query "<problem text>" [field]     show what would be injected
  python -m train.memory stats                              store size, distillers, usefulness coverage
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    cmd, args = (argv[0] if argv else "help"), argv[1:]
    if cmd == "build":
        from .builder import build
        build([a for a in args if not a.startswith("--")] or None, distill="auto" if "--llm" in args else ("extractive" if "--extractive" in args else None))
    elif cmd == "eval":
        from .eval import run
        run(llm="--no-llm" not in args)
    elif cmd == "eval-graph":
        from .eval import graph_eval
        graph_eval()
    elif cmd == "hygiene":
        from .hygiene import run as hygiene
        hygiene()
    elif cmd == "update":
        from .api import update_from_results
        print("usefulness: %d graded rows applied" % update_from_results())
    elif cmd == "serve":
        from .daemon import serve
        serve()
    elif cmd == "query":
        from . import api
        hits = api.retrieve(args[0], field=args[1] if len(args) > 1 else None)
        for s, c in hits:
            print("%.3f %s" % (s, c["id"]))
        print(api.render(hits))
    elif cmd == "stats":
        from collections import Counter
        from . import store
        cards, vecs = store.load()
        st = store.load_stats()
        print(json.dumps({"cards": len(cards), "vectors": len(vecs), "fields": Counter(c["field"] for c in cards),
                          "distiller": Counter(c["distiller"] for c in cards), "with_outcomes": len(st["cards"])}, indent=1))
    else:
        print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))