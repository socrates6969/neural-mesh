"""Verify-before-keep gate. compare(base, cand): per axis, per world, per split (HELD, CONF): paired seed diffs, Holm over all
(axis x world x split) tests of ONE candidate (family = every axis tested). Verdict:
  improves  : some TARGET axis has Holm-significant gain in the same world in BOTH HELD and CONF (and no sign flip)
  regresses : ANY axis has a Holm-significant loss (|diff| > tol) in HELD or CONF
  KEEP only if improves and not regresses.  TOST (+-delta) reported for non-target axes as equivalence evidence."""
import numpy as np
from lib import *

def axes_of(tab):
    return sorted({k for v in tab.values() for k in v if not k.startswith("us_")})

def compare(base, cand, target, tol=0.0, delta=0.01, quiet=False, axes=None):
    axes = axes or [a for a in axes_of(base) if a in axes_of(cand)]
    rows = []
    for ax in axes:
        for w in WORLDS + "R":
            for sp, seeds in (("HELD", HELD), ("CONF", CONF)):
                a = [cand[(w, s)][ax] for s in seeds if (w, s) in cand and ax in cand[(w, s)]]; b = [base[(w, s)][ax] for s in seeds if (w, s) in base and ax in base[(w, s)]]
                if ax.startswith("tok_"): a, b = [-x for x in a], [-x for x in b]  # COST axis (tokens injected per query): lower is better, so the sign is flipped (R71 first-class token axis)
                if len(a) != len(b) or len(a) < 5: continue
                st = paired(a, b); st.update(ax=ax, w=w, sp=sp, tost=tost(a, b, delta)); rows.append(st)
    ps = [r["p"] for r in rows]; adj = holm(ps) if ps else []
    for r, ph in zip(rows, adj): r["ph"] = ph
    gain, loss = [], []
    for ax in axes:
        for w in WORLDS + "R":
            h = [r for r in rows if r["ax"] == ax and r["w"] == w]
            t_ = max(tol, 0.5) if ax.startswith("tok_") else tol  # token axes: |change| must exceed 0.5 token to count
            if len(h) == 2 and all(r["ph"] < 0.05 and r["mean"] > t_ for r in h): gain.append((ax, w, [round(r["mean"], 4) for r in h]))
            for r in h:
                if r["ph"] < 0.05 and r["mean"] < -t_: loss.append((ax, w, r["sp"], round(r["mean"], 4)))
    imp = [g for g in gain if g[0] in target]
    verdict = "KEEP" if imp and not loss else ("REJECT-regress" if loss and imp else "REJECT-nogain")
    if not quiet:
        print(f"  target gains (Holm, both splits): {imp or 'none'}")
        print(f"  other gains: {[g for g in gain if g[0] not in target] or 'none'}")
        print(f"  Holm losses: {loss or 'none'}")
        for ax in target:
            for w in WORLDS + "R":
                for r in rows:
                    if r["ax"] == ax and r["w"] == w:
                        print(f"    {ax:16s} {w} {r['sp']}: d={r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] p_holm={r['ph']:.3g} W/L={r['wins']}/{r['losses']}")
        print("  VERDICT:", verdict)
    return verdict, rows, gain, loss
