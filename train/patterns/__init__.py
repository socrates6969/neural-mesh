"""Multi-agent orchestration patterns for training and eval (the mesh-* skills are the interactive versions).

best_of_n: N independent attempts, verified-majority selection, agreement as confidence.
evolve:    population search with a pluggable scorer, novelty bonus, negative memory and paradigm-shift mutation.
"""
from .best_of_n import answer_key, best_of_n  # noqa: F401
from .evolve import Candidate, combine_novelty, cosine_novelty, evolve, token_novelty, unexplored_bonus  # noqa: F401
