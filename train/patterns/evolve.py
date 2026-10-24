"""Idea evolution loop (AlphaEvolve-style skeleton) with pluggable scorer, novelty bonus, negative memory and
paradigm-shift mutation. No LLM calls here: `mutate` / `crossover` / `scorer` are callables supplied by the caller
(an LLM agent in the mesh-evolve skill, stubs in the tests).

fitness = score + novelty_weight * novelty, where novelty is the distance to the nearest item in `archive`
(existing workspace items) and the rest of the current pool. Selection uses fitness; elites are kept by raw score so
the best solution is never lost to the novelty bonus.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s):
    return set(_TOK.findall((s or "").lower()))


def token_novelty(content: str, refs) -> float:
    """1 - max Jaccard similarity to any reference (1.0 when there are no references). Cheap default."""
    a = _tokens(content)
    best = 0.0
    for r in refs:
        b = _tokens(r)
        if a or b:
            best = max(best, len(a & b) / len(a | b))
    return 1.0 - best


def cosine_novelty(embed):
    """Novelty from an embedding function (e.g. MiniLM via train.memory): 1 - max cosine to any reference."""
    cache = {}

    def vec(s):
        if s not in cache:
            cache[s] = list(embed(s))
        return cache[s]

    def cos(u, v):
        nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
        return sum(x * y for x, y in zip(u, v)) / (nu * nv) if nu and nv else 0.0

    def novelty(content, refs):
        refs = list(refs)
        if not refs:
            return 1.0
        u = vec(content)
        return 1.0 - max(cos(u, vec(r)) for r in refs)

    return novelty


def unexplored_bonus(concepts_fn, known):
    """Curiosity bonus for unexplored regions: the share of a candidate's concepts that the workspace concept graph has
    not seen yet. concepts_fn(text) -> list of concept strings (e.g. train.memory.workspace.concepts); known = set of
    concepts already in the graph. Signature matches novelty_fn (refs are ignored)."""
    known = set(known)

    def bonus(content, refs=()):
        cs = [c for c in concepts_fn(content) if not str(c).startswith("field:")]
        return sum(c not in known for c in cs) / len(cs) if cs else 0.0

    return bonus


def combine_novelty(*parts):
    """Weighted sum of novelty functions: combine_novelty((cosine_novelty(embed), 1.0), (unexplored_bonus(...), 0.5))."""
    def novelty(content, refs):
        return sum(w * f(content, refs) for f, w in parts)

    return novelty


def first_line(content: str) -> str:
    """Default key idea of a candidate: its first non-empty line (candidates should lead with their core idea)."""
    return next((ln.strip() for ln in (content or "").splitlines() if ln.strip()), "")


@dataclass(eq=False)
class Candidate:
    content: str
    score: float = float("-inf")
    novelty: float = 0.0
    fitness: float = float("-inf")
    gen: int = 0
    op: str = "seed"
    parents: tuple = ()
    meta: dict = field(default_factory=dict)


def evolve(seeds, scorer, mutate, *, crossover=None, generations: int = 5, pop_size: int = 8, elite: int = 2,
           novelty_fn=token_novelty, novelty_weight: float = 0.2, archive=(), negative=(), negative_sim: float = 0.85,
           fail_below=None, key_idea_fn=first_line, paradigm_every: int = 2, crossover_rate: float = 0.3,
           max_evals: int | None = None, target: float | None = None, patience: int | None = None, seed: int = 0,
           on_generation=None) -> dict:
    """Evolve `seeds` (list of strings) for `generations` generations.

    scorer(content) -> float (higher is better), or a dict with a "score" key; cached by content.
    mutate(parent: Candidate, ctx) -> str. ctx = {"mode": "normal"|"paradigm_shift", "banned": [key ideas],
        "negative": [past failures], "gen": g, "rng": rng}. In paradigm-shift mode the child must not reuse a
        banned idea; children whose key idea equals a banned one are rejected.
    crossover(a: Candidate, b: Candidate, ctx) -> str, optional.
    negative: past failures (e.g. from the workspace). A child within `negative_sim` similarity of one is rejected
        before scoring. Children scoring <= fail_below are added to the negative memory during the run.
    Stops early at `target`, after `patience` generations without improvement, or when `max_evals` is used up.
    """
    rng = random.Random(seed)
    archive = list(archive)
    negative = list(negative)
    cache, stats = {}, {"evals": 0, "rejected_negative": 0, "rejected_banned": 0, "rejected_duplicate": 0}

    def budget_left():
        return max_evals is None or stats["evals"] < max_evals

    def score(c):
        if c.content not in cache:
            if not budget_left():
                return False
            s = scorer(c.content)
            cache[c.content] = float(s["score"] if isinstance(s, dict) else s)
            stats["evals"] += 1
        c.score = cache[c.content]
        return True

    def near_negative(content):
        return any(1.0 - token_novelty(content, [n]) >= negative_sim for n in negative)

    def refit(pool):
        for c in pool:
            others = archive + [o.content for o in pool if o is not c]
            c.novelty = novelty_fn(c.content, others)
            c.fitness = c.score + novelty_weight * c.novelty

    def pick(pool, k=3):
        return max(rng.sample(pool, min(k, len(pool))), key=lambda c: c.fitness)

    pop = []
    for s in seeds:
        c = Candidate(s)
        if score(c):
            pop.append(c)
    refit(pop)
    if not pop:
        return {"best": None, "population": [], "history": [], "negative": negative, **stats}
    best = max(pop, key=lambda c: c.score)
    history = [{"gen": 0, "best": best.score, "mean": sum(c.score for c in pop) / len(pop), "n": len(pop)}]
    stale = 0
    seen = {c.content for c in pop}

    for g in range(1, generations + 1):
        if (target is not None and best.score >= target) or not budget_left():
            break
        shift = bool(paradigm_every) and g % paradigm_every == 0
        banned = [key_idea_fn(best.content)] if shift else []
        n_shift = max(1, (pop_size - elite) // 2) if shift else 0
        children, tries = [], 0
        while len(children) < pop_size - elite and tries < pop_size * 4 and budget_left():
            tries += 1
            mode = "paradigm_shift" if len(children) < n_shift else "normal"
            ctx = {"mode": mode, "banned": banned if mode == "paradigm_shift" else [], "negative": negative,
                   "gen": g, "rng": rng}
            if mode == "normal" and crossover and len(pop) >= 2 and rng.random() < crossover_rate:
                a, b = pick(pop), pick(pop)
                child = Candidate(crossover(a, b, ctx), gen=g, op="crossover", parents=(a.content, b.content))
            else:
                p = pick(pop)
                child = Candidate(mutate(p, ctx), gen=g, op=mode, parents=(p.content,))
            if not child.content or child.content in seen:
                stats["rejected_duplicate"] += 1
                continue
            seen.add(child.content)
            if mode == "paradigm_shift" and key_idea_fn(child.content) in banned:
                stats["rejected_banned"] += 1
                continue
            if near_negative(child.content):
                stats["rejected_negative"] += 1
                continue
            if score(child):
                children.append(child)
                if fail_below is not None and child.score <= fail_below:
                    negative.append(child.content)
        pool = pop + children
        refit(pool)
        elites = sorted(pool, key=lambda c: c.score, reverse=True)[:elite]
        rest = sorted((c for c in pool if c not in elites), key=lambda c: c.fitness, reverse=True)
        pop = elites + rest[:max(0, pop_size - len(elites))]
        top = max(pop, key=lambda c: c.score)
        if top.score > best.score:
            best, stale = top, 0
        else:
            stale += 1
        history.append({"gen": g, "best": best.score, "mean": sum(c.score for c in pop) / len(pop),
                        "n": len(children), "paradigm_shift": shift, "banned": banned})
        if on_generation:
            on_generation(g, pop, best)
        if patience is not None and stale >= patience:
            break
    return {"best": best, "population": pop, "history": history, "negative": negative, **stats}
