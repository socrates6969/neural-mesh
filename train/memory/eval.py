"""Offline retrieval eval: for held-out benchmark problems, is a card of the same subfield / method retrieved?

Labels (relevance = query and card share the label; queries without a relevant card in the corpus are dropped):
* math            Omni-MATH `domain`, level 2 (e.g. "Discrete Mathematics -> Combinatorics"); MathArena rows unlabelled
* coding          Codeforces problem tags (problemset API), sharing at least one method tag; generic tags
                  (implementation, math, greedy, brute force, constructive algorithms, sortings, *special) do not count
* neural-networks generator template of the problem (`source`, e.g. generated-hard:kv)
* cybersecurity   the gold CWE id
* physics, computer-science (secondary, weaker): a topic from a fixed taxonomy assigned by the local model
Strong labels come from the datasets themselves, so they do not favour any retriever; the LLM topics are reported
separately. Scores: hit@1/3/5, MRR@10, plus the random-ranking expectation of hit@k for scale.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict

import numpy as np

from . import config as C
from . import search, store

GENERIC_TAGS = {"implementation", "math", "greedy", "brute force", "constructive algorithms", "sortings", "*special"}
TOPICS = {
    "physics": ["kinematics and Newtonian mechanics", "work, energy and momentum", "rotational mechanics",
                "gravitation", "fluids and elasticity", "thermodynamics and kinetic theory", "waves, sound and oscillations",
                "geometrical and wave optics", "electrostatics and capacitors", "current electricity and circuits",
                "magnetism and electromagnetic induction", "modern and nuclear physics"],
    "computer-science": ["string manipulation", "list and array processing", "math and number formulas",
                         "searching and sorting", "dictionaries, sets and counting", "recursion and dynamic programming",
                         "regular expressions", "geometry formulas", "bit manipulation", "tuples and nested data"],
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _bench(field: str) -> list[dict]:
    p = C.RUNS / "bench" / (field + ".jsonl")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def strong_labels() -> dict:
    """task id -> frozenset of labels, from dataset metadata."""
    lab = {}
    om = [json.loads(l) for l in (C.LABELS / "omni_math_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    heads = [(_norm(o["problem"])[:120], o) for o in om]
    for t in _bench("math"):
        p = _norm(t["prompt"])
        for h, o in heads:
            if h and h in p:
                lab[t["id"]] = frozenset(" -> ".join(d.split(" -> ")[1:3]) for d in o["domain"])
                break
    cf = json.loads((C.LABELS / "cf_problemset.json").read_text(encoding="utf-8"))["result"]["problems"]
    tags = {"cc-%s%s" % (p["contestId"], p["index"]): set(p["tags"]) - GENERIC_TAGS for p in cf if "contestId" in p}
    for t in _bench("coding"):
        if tags.get(t["id"]):
            lab[t["id"]] = frozenset(tags[t["id"]])
    for t in _bench("neural-networks"):
        lab[t["id"]] = frozenset([t["source"]])
    for t in _bench("cybersecurity"):
        if t.get("answer"):
            lab[t["id"]] = frozenset([str(t["answer"]).strip().upper()])
    return lab


def llm_labels(fields=("physics", "computer-science"), log=print) -> dict:
    """task id -> frozenset([topic]) from the local model (cached in labels/llm_topics.json)."""
    from .cards import _chat
    path = C.LABELS / "llm_topics.json"
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for f in fields:
        opts = TOPICS[f]
        for t in _bench(f):
            if t["id"] in cache:
                continue
            sysmsg = ("Classify the problem into exactly one topic from this list: " + "; ".join(opts) +
                      '. Reply with JSON {"topic": "<one topic copied exactly from the list>"}.')
            d = _chat(sysmsg, t["prompt"][:2500])
            top = (d or {}).get("topic", "")
            if top in opts:
                cache[t["id"]] = top
            if len(cache) % 50 == 0:
                path.write_text(json.dumps(cache), encoding="utf-8")
                log("  llm labels: %d" % len(cache))
    path.write_text(json.dumps(cache), encoding="utf-8")
    return {k: frozenset([v]) for k, v in cache.items()}


def _texts(cards, variant):
    if variant == "problem":
        return [c.get("problem") or "" for c in cards]
    if variant == "card":
        return [c.get("text") or "" for c in cards]
    return [store.index_text(c) for c in cards]


def ranking(idx: search.Index, mode: str, query: str, qvec, avail, ce=None, depth: int = 10):
    """Top-`depth` card indices for one query (no gate) and their stage scores.
    mode: dense | bm25 | hybrid | hybrid@W (dense weight W) | rerank (CE over hybrid top) | dense_rerank (CE over dense top)."""
    if mode in ("rerank", "dense_rerank"):
        base = "hybrid" if mode == "rerank" else "dense"
        cand = idx.candidates(query, qvec, avail, base)[:C.RERANK_TOP if mode == "rerank" else 10]
        if not cand:
            return [], []
        sc = ce.score(query[:C.MAX_PROBLEM], [idx.texts[i] for i, _, _ in cand])
        order = np.argsort(-sc)
        return [cand[j][0] for j in order][:depth], [float(sc[j]) for j in order][:depth]
    w = 1.0
    if mode.startswith("hybrid@"):
        mode, w = "hybrid", float(mode.split("@")[1])
    cand = idx.candidates(query, qvec, avail, mode, dense_w=w)
    if mode == "hybrid":
        return [i for i, _, _ in cand][:depth], [c for _, _, c in cand][:depth]
    return [i for i, _, _ in cand][:depth], [s for _, s, _ in cand][:depth]


def _rand_hit(n: int, r: int, k: int) -> float:
    if n <= 0 or r <= 0:
        return 0.0
    k = min(k, n)
    return 1.0 - math.comb(n - r, k) / math.comb(n, k) if n - r >= k else 1.0


def _sample(tasks: dict, max_per_field: int) -> list[str]:
    """Deterministic cap per field (sha1 order of the task id), so large label sets do not dominate the eval."""
    import hashlib
    by = defaultdict(list)
    for q, t in tasks.items():
        by[t["field"]].append(q)
    out = []
    for f, qs in by.items():
        out += sorted(qs, key=lambda q: hashlib.sha1(q.encode()).hexdigest())[:max_per_field]
    return sorted(out)


def evaluate(labels: dict, modes=("dense", "bm25", "hybrid", "rerank"), variants=("both",), field_filter=True,
             max_per_field: int = 200, log=print, ce_dirs: dict | None = None) -> dict:
    from .. import experience
    from . import rerank
    cards, _ = store.load()
    enc = experience.encoder()
    ces = {name: rerank.CrossEncoder(d) for name, d in (ce_dirs or {}).items()} if ce_dirs else {"": rerank.get()}
    tasks = {}
    for f in {c["field"] for c in cards}:
        for t in _bench(f):
            if t["id"] in labels:
                tasks[t["id"]] = t
    qids = _sample(tasks, max_per_field)
    qvecs = enc.encode([tasks[q]["prompt"][:C.MAX_PROBLEM] for q in qids])
    card_lab = [labels.get(c["task"], frozenset()) for c in cards]
    results, calib, firsts = {}, defaultdict(list), defaultdict(dict)
    for variant in variants:
        texts = _texts(cards, variant)
        idx = search.Index(cards, enc.encode(texts), texts)
        for mode_full in modes:
            mode, _, cename = mode_full.partition(":")
            ce = ces.get(cename) if mode.endswith("rerank") else None
            if mode.endswith("rerank") and ce is None:
                continue
            per = defaultdict(lambda: defaultdict(list))
            t0 = time.perf_counter()
            for qi, q in enumerate(qids):
                t = tasks[q]
                avail = idx.mask(t["field"] if field_filter else None, {q})
                rel = {i for i in np.flatnonzero(avail) if card_lab[i] & labels[q]}
                if not rel:
                    continue
                order, scores = ranking(idx, mode, t["prompt"], qvecs[qi], avail, ce)
                first = next((r for r, i in enumerate(order, 1) if i in rel), None)
                firsts[(variant, mode_full)][q] = first or 0
                m = per[t["field"]]
                for k in (1, 3, 5):
                    m["hit@%d" % k].append(1.0 if first and first <= k else 0.0)
                    m["rand@%d" % k].append(_rand_hit(int(avail.sum()), len(rel), k))
                m["mrr"].append(1.0 / first if first else 0.0)
                if variant == variants[0] and scores:
                    calib[mode_full].append((scores[0], order[0] in rel, t["field"]))
            ms = (time.perf_counter() - t0) / max(1, len(qids)) * 1000
            rows = {f: {k: round(float(np.mean(v)), 3) for k, v in m.items()} | {"n": len(m["mrr"])} for f, m in per.items()}
            allm = defaultdict(list)
            for m in per.values():
                for k, v in m.items():
                    allm[k].extend(v)
            rows["ALL"] = {k: round(float(np.mean(v)), 3) for k, v in allm.items()} | {"n": len(allm["mrr"])}
            exnn = [v for f, m in per.items() if f != "neural-networks" for v in m["mrr"]]
            exh1 = [v for f, m in per.items() if f != "neural-networks" for v in m["hit@1"]]
            rows["ALL_exNN"] = {"hit@1": round(float(np.mean(exh1)), 3), "mrr": round(float(np.mean(exnn)), 3), "n": len(exnn)}
            results["%s/%s" % (variant, mode_full)] = {"fields": rows, "ms_per_query": round(ms, 1),
                                                       "first": firsts[(variant, mode_full)]}
            log("%-8s %-22s n=%-4d hit@1 %.3f hit@3 %.3f hit@5 %.3f MRR %.3f | exNN hit@1 %.3f MRR %.3f | %.1f ms/q" % (
                variant, mode_full, rows["ALL"]["n"], rows["ALL"]["hit@1"], rows["ALL"]["hit@3"], rows["ALL"]["hit@5"],
                rows["ALL"]["mrr"], rows["ALL_exNN"]["hit@1"], rows["ALL_exNN"]["mrr"], ms))
    results["calibration"] = {m: calibrate(v) for m, v in calib.items()}
    results["calibration_exNN"] = {m: calibrate([p for p in v if p[2] != "neural-networks"]) for m, v in calib.items()}
    return results


def calibrate(pairs: list) -> dict:
    """Gate on the top-1 stage score: the threshold maximising F0.5 of 'injected card is relevant' (precision-weighted),
    with precision and coverage at that threshold."""
    if not pairs:
        return {}
    xs = sorted(pairs, key=lambda p: -p[0])
    total_rel = sum(1 for _, r, _ in xs if r)
    best, tp = None, 0
    for n, (s, r, _) in enumerate(xs, 1):
        tp += r
        p, rc = tp / n, tp / max(1, total_rel)
        f = 1.25 * p * rc / (0.25 * p + rc) if p + rc else 0.0
        if best is None or f > best["f05"]:
            best = {"gate": round(s, 4), "f05": round(f, 3), "precision": round(p, 3), "coverage": round(n / len(xs), 3)}
    best["base_precision"] = round(total_rel / len(xs), 3)
    return best


MODES = ("dense", "bm25", "hybrid", "hybrid@2", "dense_rerank", "rerank")


def run(llm: bool = True, log=print) -> dict:
    """Full eval. llm=True adds the LLM-topic labels (needs the local Ollama model - only when the GPU is free)."""
    nn = {t["id"]: f != "neural-networks" for f in ("math", "coding", "cybersecurity", "neural-networks") for t in _bench(f)}
    r = evaluate(strong_labels(), modes=MODES, variants=("both", "problem", "card"), log=log)
    r["paired_vs_dense_exNN"] = {m: paired(r, "both/dense", "both/" + m, nn) for m in MODES[1:] if "both/" + m in r}
    out = {"strong": r}
    log("-- strong labels, variants x modes done --")
    if llm:
        log("-- secondary: LLM topic labels (physics, computer-science) --")
        out["llm"] = evaluate(llm_labels(log=log), log=log)
    log("-- strong labels, no field filter (hook setting) --")
    out["strong_nofield"] = evaluate(strong_labels(), modes=MODES[:4], log=log, field_filter=False)
    p = C.MEM_DIR / "eval.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log("wrote " + str(p))
    return out

def paired(results: dict, a: str, b: str, labels_field: dict | None = None, n_boot: int = 5000, seed: int = 0) -> dict:
    """Paired bootstrap of the MRR difference b - a over the same queries (optionally excluding some fields)."""
    fa, fb = results[a]["first"], results[b]["first"]
    qs = [q for q in fa if q in fb and (labels_field is None or labels_field.get(q, True))]
    da = np.array([1.0 / fa[q] if fa[q] else 0.0 for q in qs])
    db = np.array([1.0 / fb[q] if fb[q] else 0.0 for q in qs])
    d = db - da
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"n": len(qs), "diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(boots, 2.5)), 4),
            round(float(np.percentile(boots, 97.5)), 4)], "p_le0": round(float((boots <= 0).mean()), 4)}
