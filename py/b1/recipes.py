"""Opt-in scoring recipes verified by the R&D gate (python reference; no Rust port needed - they are 2-line formulas over cosine similarities).
cloze_mix (R38): pick argmax_o cos(E(stem), E(stem with o filled in)) + 0.5 * cos(E(stem), E(o)).
CAVEAT (R38): the gain comes from topic association between the stem and the option; on a hard set where ALL options are on-topic it is
chance-level either way (see py/b1/r38h.py)."""
import numpy as np
def _cos(a, B): return (B @ a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1) + 1e-9)
def cloze_mix(embed, stem, options, lam=0.5):
    F = np.stack([embed(stem.replace("___", o)) for o in options]); O = np.stack([embed(o) for o in options]); s = embed(stem)
    return int(np.argmax(_cos(s, F) + lam * _cos(s, O)))
