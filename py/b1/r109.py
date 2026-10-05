"""R109: formal Holm + TOST on the Rust-encoder retrieval equivalence (results/r101_ranks.json from src/bin/n17_encoder_retrieval.rs; tie-pessimistic ranks).
Per-query paired differences vs the stored-ORT-fp32-query base, item-disjoint halves (HELD = even query index, CONF = odd). Holm over all difference tests; TOST +-.01 (family Holm over the cells' max one-sided p)."""
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
            for sp, off in (("HELD", 0), ("CONF", 1)):
                a = [f(r) for r in R[c][var][off::2]]; b = [f(r) for r in base[off::2]]
                st = paired(a, b); st.update(c=c, var=var, m=m, sp=sp, tost=tost(a, b, 0.01)); rows.append(st)
ph = holm([r["p"] for r in rows]); pt = holm([r["tost"] for r in rows])
for r, p1, p2 in zip(rows, ph, pt): r["ph"], r["pt"] = p1, p2
print("corpus variant metric split   n   diff    90%CI-ish(95%)        p_holm  TOST_p  TOST_holm  W/L")
for r in rows: print(f"{r['c']} {r['var']:8s} {r['m']:5s} {r['sp']:4s} {r['n']:4d} {r['mean']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['ph']:.3g} {r['tost']:.2g} {r['pt']:.2g} {r['wins']}/{r['losses']}")
print("max |diff|:", max(abs(r["mean"]) for r in rows), "| diff Holm-significant cells:", [(r['c'], r['var'], r['m'], r['sp'], round(r['mean'], 4)) for r in rows if r["ph"] < .05], "| max TOST Holm p:", max(r["pt"] for r in rows))
