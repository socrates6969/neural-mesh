"""R115: POOLED (HELD+CONF halves merged, full sample) Holm + TOST of the Rust-encoder retrieval equivalence; settles R112's one inconclusive cell (B int8 hit@5 CONF).
Same data as r109.py (results/r101_ranks.json), same paired stats/TOST margin +-.01; the halves are NOT independent confirmation any more (pooled = one full-sample test per cell)."""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import paired, holm, tost
R = json.load(open(os.path.join(os.path.dirname(__file__), "../../results/r101_ranks.json")))
metric = {"RR": lambda r: 0.0 if r < 0 else 1.0 / (r + 1), "hit1": lambda r: float(r == 0), "hit5": lambda r: float(0 <= r < 5)}
rows = []
for c in "AB":
    base = R[c]["base"]
    for var in ("rust_f32", "rust_i8"):
        for m, f in metric.items():
            a = [f(r) for r in R[c][var]]; b = [f(r) for r in base]
            st = paired(a, b); st.update(c=c, var=var, m=m, tost=tost(a, b, 0.01)); rows.append(st)
ph = holm([r["p"] for r in rows]); pt = holm([r["tost"] for r in rows])
for r, p1, p2 in zip(rows, ph, pt): r["ph"], r["pt"] = p1, p2
print("corpus variant metric   n   diff    95%CI            p_holm  TOST_p  TOST_holm  W/L")
for r in rows: print(f"{r['c']} {r['var']:8s} {r['m']:5s} {r['n']:4d} {r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['ph']:.3g} {r['tost']:.2g} {r['pt']:.2g} {r['wins']}/{r['losses']}")
print("max |diff|:", max(abs(r["mean"]) for r in rows), "| diff Holm-significant:", [(r['c'], r['var'], r['m']) for r in rows if r["ph"] < .05], "| max TOST Holm p:", max(r["pt"] for r in rows))
