"""EXPERIMENT, NOT WIRED IN: `python -m train.memory eval-graph` (strong-label held-out queries, own card excluded,
leave-one-out graph) found graph re-ranking WORSE in every field and variant - paired MRR vs plain hybrid -0.034 to
-0.061 (95% CIs exclude 0); precision at the per-field gates: math .655 -> .48-.50, coding .453 -> .37-.41,
cybersecurity .846 -> .75-.79. The keyword/bigram concepts are too noisy for expansion to help. Kept for re-testing if
the concept extractor improves (e.g. LLM-distilled method tags).

Graph-RAG over the concept co-occurrence graph: expand a query's concepts with their 1-2 hop neighbours and use
the overlap with a candidate's concepts to re-rank retrieval candidates.

  expanded = {query concepts: 1.0} + hop-1 neighbours (0.5 x normalised edge weight) + hop-2 (0.25 x ...)
  bonus(candidate) = sum of expanded weights of the candidate's concepts / sum of all expanded weights   (0..1)
  re-ranked score = fused score x (1 + LAMBDA x bonus)

Two graph sources share one interface (neighbours(concept, per) -> [(neighbour, weight)]):
  WorkspaceGraph  the live workspace `edges` table (production)
  CardGraph       built from the card store's concepts, with leave-one-out (offline eval: a held-out query's own
                  card must not shape its own expansion)
"""
from __future__ import annotations

from collections import Counter, defaultdict

LAMBDA = 0.5
HOP_W = (1.0, 0.5, 0.25)


def card_concepts(card: dict) -> list[str]:
    from .workspace import concepts
    return [c for c in concepts((card.get("problem") or "") + "\n" + (card.get("text") or "")) if not c.startswith("field:")]


class CardGraph:
    def __init__(self, cards: list[dict]):
        self.cc = [set(card_concepts(c)) for c in cards]
        self.w = defaultdict(Counter)
        for cs in self.cc:
            for a in cs:
                for b in cs:
                    if a != b:
                        self.w[a][b] += 1
        self.skip = None                                    # card index left out (leave-one-out)

    def leave_out(self, i):
        self.skip = i

    def neighbours(self, c: str, per: int = 5):
        cnt = self.w.get(c)
        if not cnt:
            return []
        own = self.cc[self.skip] if self.skip is not None else ()
        items = []
        for n, w in cnt.items():
            if self.skip is not None and c in own and n in own:
                w -= 1                                      # remove the left-out card's co-occurrence
            if w > 0:
                items.append((n, w))
        items.sort(key=lambda x: (-x[1], x[0]))
        return items[:per]


class WorkspaceGraph:
    def __init__(self, db=None):
        self.db = db
        self._cache = {}

    def neighbours(self, c: str, per: int = 5):
        if c in self._cache:
            return self._cache[c][:per]
        from . import workspace as W
        con = W._connect(self.db)
        try:
            rows = con.execute("SELECT b AS n, w FROM edges WHERE a = ? UNION ALL SELECT a AS n, w FROM edges WHERE b = ? "
                               "ORDER BY w DESC LIMIT ?", (c, c, max(per, 10))).fetchall()
        finally:
            con.close()
        out = [(r["n"], r["w"]) for r in rows if not r["n"].startswith("field:")]
        self._cache[c] = out
        return out[:per]


def expand(query_concepts, graph, hops: int = 1, per: int = 5) -> dict:
    exp = {c: HOP_W[0] for c in query_concepts}
    frontier = list(query_concepts)
    for h in range(1, hops + 1):
        nxt = []
        for c in frontier:
            nb = graph.neighbours(c, per)
            top = max((w for _, w in nb), default=1)
            for n, w in nb:
                v = HOP_W[h] * w / top
                if v > exp.get(n, 0.0):
                    exp[n] = v
                    nxt.append(n)
        frontier = nxt
    return exp


def bonus(expanded: dict, cand: set) -> float:
    tot = sum(expanded.values())
    return sum(v for c, v in expanded.items() if c in cand) / tot if tot else 0.0


def rerank(cand: list, expanded: dict, cand_concepts, lam: float = LAMBDA) -> list:
    """cand: [(index, fused score, cosine)] -> same tuples re-ordered by fused x (1 + lam x bonus)."""
    scored = [(f * (1 + lam * bonus(expanded, cand_concepts(i))), i, f, c) for i, f, c in cand]
    scored.sort(key=lambda x: -x[0])
    return [(i, f, c) for _, i, f, c in scored]