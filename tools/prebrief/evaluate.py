"""Offline evaluation O1 (pre-registered in ruv/hive/findings/prebrief.md): prequential, time-split hit@k of the brief retriever.

python -m prebrief.evaluate --corpus <corpus.jsonl> --out <dir> [--include-session]
"""
import argparse
import json
import math
import os
import time

import numpy as np

from .corpus import load
from .index import Index, anchors, relevant


def query_text(d: dict) -> str:
    return f"{d['title']}\n{d['prompt'][:3000]}\n{d['agent']}"


def binom_tail_ge(n: int, k: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n)


def mde_sign(m: int, power: float = 0.8, alpha: float = 0.05) -> float | None:
    """Smallest win share among m decisive pairs detectable with `power` by a two-sided exact sign test at `alpha`."""
    if m < 1:
        return None
    crit = next((k for k in range(m + 1) if binom_tail_ge(m, k, 0.5) <= alpha / 2), None)
    if crit is None:
        return None
    for p in [x / 1000 for x in range(500, 1001)]:
        if binom_tail_ge(m, crit, p) >= power:
            return p
    return None


def ranks(scores: np.ndarray, rel: set) -> int | None:
    order = np.argsort(-scores, kind="stable")
    for r, i in enumerate(order[:10]):
        if not np.isfinite(scores[i]):
            break
        if int(i) in rel:
            return r + 1
    return None


def run(corpus_path: str, out_dir: str, include_session: bool = False, alphas=(0.3, 0.5, 0.7), label: str = "report"):
    docs = load(corpus_path)
    idx = Index(docs, cache_dir=os.path.join(out_dir, "vec_cache"))
    t0 = time.time()
    have_vec = idx.ensure_vectors()
    embed_s = time.time() - t0
    if label == "report":
        qs = [i for i, d in enumerate(docs) if d["kind"] == "run" and len(d.get("report", "")) > 200]
    else:  # exploratory E1: what the agent touched, minus what the task already named
        qs = [i for i, d in enumerate(docs) if d["kind"] == "run" and d.get("trace_anchors")]
    n = len(qs)
    test = qs[int(n * 0.75):]
    dev = qs[int(n * 0.60): int(n * 0.75)]
    qvecs = idx.embed([query_text(docs[i]) for i in qs]) if have_vec else None
    qvec_of = {i: qvecs[j] for j, i in enumerate(qs)} if qvecs is not None else {}

    def score_set(ids, alpha):
        rows = []
        for i in ids:
            q = docs[i]
            avail = idx.available(q["ts"], None if include_session else q["session"])
            qa = None if label == "report" else set(q["trace_anchors"]) - anchors(q["prompt"])
            rel = relevant(idx, q["report"], avail, qa=qa)
            row = {"id": q["id"], "ts": q["ts"], "agent": q["agent"], "n_rel": len(rel), "n_avail": int(avail.sum())}
            if rel:
                qt = query_text(q)
                for m in ("R", "B", "P"):
                    s = idx.search(qt, q["agent"], avail, method=m, alpha=alpha, qvec=qvec_of.get(i))
                    row[m] = ranks(s, rel)
            rows.append(row)
        return rows

    def summary(rows):
        cov = [r for r in rows if r["n_rel"] > 0]
        out = {"n_queries": len(rows), "n_covered": len(cov), "coverage": round(len(cov) / max(1, len(rows)), 3)}
        for m in ("R", "B", "P"):
            rk = [r[m] for r in cov]
            out[m] = {"hit@1": round(sum(1 for x in rk if x == 1) / max(1, len(rk)), 3),
                      "hit@5": round(sum(1 for x in rk if x and x <= 5) / max(1, len(rk)), 3),
                      "mrr@10": round(sum(1 / x for x in rk if x) / max(1, len(rk)), 3)}
        for a, bm in (("P", "B"), ("P", "R"), ("B", "R")):
            h = lambda r, m: bool(r[m] and r[m] <= 5)
            b = sum(1 for r in cov if h(r, a) and not h(r, bm))
            c = sum(1 for r in cov if h(r, bm) and not h(r, a))
            out[f"{a}_vs_{bm}_hit5"] = {"a_only": b, "b_only": c, "mcnemar_p": round(mcnemar_exact(b, c), 5), "mde_win_share": mde_sign(b + c)}
        return out

    dev_res = {a: summary(score_set(dev, a)) for a in alphas} if have_vec else {}
    best_alpha = max(alphas, key=lambda a: (dev_res[a]["P"]["hit@5"], dev_res[a]["P"]["mrr@10"])) if dev_res else None
    test_rows = score_set(test, best_alpha or 0.5)
    res = {"label": label, "setting": ("include_session" if include_session else "exclude_session") + ("" if label == "report" else "_" + label), "n_docs": len(docs), "n_runs_with_report": n,
           "dev": {"range": [docs[dev[0]]["ts"], docs[dev[-1]]["ts"]], "by_alpha": dev_res}, "alpha": best_alpha,
           "test": {"range": [docs[test[0]]["ts"], docs[test[-1]]["ts"]], **summary(test_rows)}, "embed_s": round(embed_s, 1),
           "dense": have_vec}
    os.makedirs(out_dir, exist_ok=True)
    tag = res["setting"]
    with open(os.path.join(out_dir, f"o1_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, f"o1_{tag}_rows.jsonl"), "w") as f:
        for r in test_rows:
            f.write(json.dumps(r) + "\n")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-session", action="store_true")
    ap.add_argument("--label", default="report", choices=["report", "trace"])
    a = ap.parse_args()
    print(json.dumps(run(a.corpus, a.out, a.include_session, label=a.label), indent=1))