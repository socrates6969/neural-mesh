"""Hybrid retrieval over the card store: BM25 (prebrief tokenizer, k1 .9 b .5) + MiniLM dense cosine, fused with
reciprocal-rank fusion, then the top RERANK_TOP re-scored by a cross-encoder. Field-aware filter, calibrated gate,
usefulness prior, at most k cards. Modes (also used by the offline eval): dense | bm25 | hybrid | rerank."""
from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import config as C

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from tools.prebrief.index import tokens  # noqa: E402  (shared tokenizer: stopwords, camel/snake split, hex filter)

# Defaults chosen by the offline eval (python -m train.memory eval; 553 strong-label queries, see eval.py):
#   dense MRR .725, hybrid (RRF 1:1) .719, hybrid dense-weighted 2:1 .728, bm25 .694, CE rerank .661/.638 (worse,
#   paired 95% CI excludes 0, and ~0.4-0.9 s/query). Dense-weighted RRF is the default: best point estimate in both
#   the field-filtered and the no-field (hook) setting, statistically tied with dense, higher precision at its gate.
# Gates: the minimum stage score a card needs to be injected. For dense/hybrid the per-field FIELD_GATES below apply
#   (MiniLM cosine); GATES holds bm25 (BM25 per distinct query term) and rerank (cross-encoder logit), used only when
#   those modes are selected explicitly, plus a pooled dense/hybrid fallback.
GATES = {"dense": 0.65, "bm25": 2.15, "hybrid": 0.65, "rerank": 7.4}
# The gate is PER FIELD: score scales differ a lot between fields. Calibrated (F0.5, precision-weighted) on full
# benchmark problems, hybrid 2:1, top-1 card, field-filtered; (gate, measured precision, coverage, base precision):
#   math .55 (.68, .26, .47)   coding .54 (.45, .46, .34)   cybersecurity .65 (.85, .88, .78)
#   physics / computer-science have no strong labels -> the pooled excl.-NN value .55 (.63, .50); NN is trivially easy.
# (A single global .65 starved math: 2.5% coverage.) Without a field, the field of the best-matching card is used.
# Queries under MIN_WORDS words are titles, not problems: problems cut to 10-15 words reach only .15-.27 precision at
# any gate, so they return nothing unless the caller opts into low_confidence (gate .40, flagged band "low").
FIELD_GATES = {"math": (0.55, 0.68), "coding": (0.54, 0.45), "cybersecurity": (0.65, 0.85), "neural-networks": (0.65, 1.0)}
DEFAULT_FIELD_GATE = (0.55, 0.63)
MIN_WORDS = 15
LOW_GATE, LOW_PRECISION = 0.40, 0.25
DEFAULT_MODE = "hybrid"
HYBRID_DENSE_W = 2.0


def gate_for(query: str, mode: str, low_confidence: bool = False, field: str | None = None):
    """(gate, measured precision, band) for this query and field, or None when the query is too short and
    low_confidence=False. band: high (precision >= .6) | medium | low (short query, explicitly requested)."""
    if mode not in ("dense", "hybrid"):
        return GATES[mode], None, "high"
    if len((query or "").split()) < MIN_WORDS:
        return (LOW_GATE, LOW_PRECISION, "low") if low_confidence else None
    g, prec = FIELD_GATES.get(field or "", DEFAULT_FIELD_GATE)
    return g, prec, "high" if prec >= 0.6 else "medium"


def rrf(rankings: list[list[int]], kk: int = C.RRF_K, weights: list[float] | None = None) -> dict[int, float]:
    """(Weighted) reciprocal-rank fusion: sum over rankings of w / (kk + rank), rank starting at 1."""
    out: dict[int, float] = defaultdict(float)
    for j, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[j]
        for pos, i in enumerate(ranking, 1):
            out[i] += w / (kk + pos)
    return dict(out)


class Index:
    def __init__(self, cards: list[dict], vecs: np.ndarray, texts: list[str] | None = None):
        from .store import index_text
        self.cards, self.vecs = cards, vecs
        self.texts = texts if texts is not None else [index_text(c) for c in cards]
        self.tf = [Counter(tokens(t)) for t in self.texts]
        self.dl = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, n in c.items():
                self.post[t].append((i, n))
        self.field = np.array([c.get("field") or "" for c in cards])
        self.task = [c.get("task") for c in cards]

    def mask(self, field: str | None, exclude: set | None) -> np.ndarray:
        m = np.ones(len(self.cards), dtype=bool)
        if field:
            m &= (self.field == field) | (self.field == "")
        if exclude:
            m &= np.array([t not in exclude for t in self.task], dtype=bool)
        return m

    def bm25(self, qtoks: list[str], avail: np.ndarray, k1: float = 0.9, b: float = 0.5) -> np.ndarray:
        s = np.zeros(len(self.cards))
        n = int(avail.sum())
        if n == 0:
            return s
        avgdl = float(self.dl[avail].mean()) or 1.0
        for t, qn in Counter(qtoks).items():
            pl = [(i, f) for i, f in self.post.get(t, ()) if avail[i]]
            if not pl:
                continue
            idf = math.log(1 + (n - len(pl) + 0.5) / (len(pl) + 0.5))
            w = 1 + 0.3 * (qn - 1)
            for i, f in pl:
                s[i] += w * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / avgdl))
        return s

    def candidates(self, query: str, qvec, avail: np.ndarray, mode: str, top: int = C.RERANK_TOP,
                   dense_w: float = 1.0):
        """[(index, stage score, cosine)] best first, for the first stage of `mode` (dense | bm25 | hybrid)."""
        idx = np.flatnonzero(avail)
        if len(idx) == 0:
            return []
        cos = self.vecs @ qvec if qvec is not None and len(self.vecs) else np.zeros(len(self.cards))
        if mode == "dense":
            return [(int(i), float(cos[i]), float(cos[i])) for i in idx[np.argsort(-cos[idx])]]
        qt = tokens(query)
        bm = self.bm25(qt, avail)
        if mode == "bm25":
            per = max(1, len(set(qt)))
            return [(int(i), float(bm[i]) / per, float(cos[i])) for i in idx[np.argsort(-bm[idx])] if bm[i] > 0]
        depth = max(top * 3, 50)
        dense_rank = [int(i) for i in idx[np.argsort(-cos[idx])][:depth]]
        lex_rank = [int(i) for i in idx[np.argsort(-bm[idx])][:depth] if bm[i] > 0]
        fused = rrf([dense_rank, lex_rank], weights=[dense_w, 1.0])
        return [(i, fused[i], float(cos[i])) for i in sorted(fused, key=lambda i: -fused[i])]

    def search(self, query: str, qvec, field: str | None = None, k: int = 2, exclude: set | None = None,
               mode: str = DEFAULT_MODE, ce=None, gate: float | None = None, stats: dict | None = None,
               top: int = C.RERANK_TOP, low_confidence: bool = False) -> list[tuple[float, dict]]:
        """Up to k (score, card), gated (length-aware unless `gate` is given). Each returned card dict is a shallow
        copy with "band" (high|medium|low) and "expected_precision" (measured top-1 precision for that band).
        Mode 'rerank' without a cross-encoder degrades to 'hybrid'."""
        if mode == "rerank" and ce is None:
            mode = "hybrid"
        band, prec = "high", None
        if gate is None and gate_for(query, mode, low_confidence, field) is None:
            return []
        avail = self.mask(field, exclude)
        cand = self.candidates(query, qvec, avail, "hybrid" if mode == "rerank" else mode, top, dense_w=HYBRID_DENSE_W)
        if gate is None:
            gf = field
            if gf is None and cand:                       # infer the field from the closest card (dense evidence)
                gf = self.cards[max(cand[:50], key=lambda x: x[2])[0]].get("field")
            gate, prec, band = gate_for(query, mode, low_confidence, gf)
        if mode == "rerank":
            cand = cand[:top]
            if not cand:
                return []
            sc = ce.score(query[:C.MAX_PROBLEM], [self.texts[i] for i, _, _ in cand])
            scored = sorted(((float(s), i) for s, (i, _, _) in zip(sc, cand)), key=lambda x: -x[0])
        elif mode == "hybrid":
            scored = [(c, i) for i, _, c in cand]          # fused order; gated by its dense evidence
        else:
            scored = [(s, i) for i, s, _ in cand]
        out = []
        for s, i in scored:
            if len(out) >= k:
                break
            if s < gate:
                if mode == "hybrid":
                    continue                              # the fused list is not sorted by cosine
                break
            card = self.cards[i]
            if stats is not None:
                from .store import usefulness
                u, n = usefulness(stats, card["id"])
                if n >= 5 and u < 0.25:                   # repeatedly in the prompt of failed tasks: stop broadcasting
                    continue
            out.append((s, dict(card, band=band, expected_precision=prec)))
        return out