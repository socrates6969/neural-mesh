"""R51 rounded-claim items (python Fractions = independent ground truth). Per (world, seed): 24 specs; each spec gets claims of kinds
exact (rational string, still must accept), round2/round3 (correct roundings, half-up), near (correct rounding +-1 in the last digit), trunc (truncation when it differs from rounding), gross (x1.5)."""
import json, os, random
from fractions import Fraction as F
from decimal import Decimal, ROUND_HALF_UP
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
U = {"m": F(1), "km": F(1000), "cm": F(1, 100), "ft": F(3048, 10000), "yd": F(9144, 10000), "in": F(254, 10000), "mile": F(1609344, 1000), "g": F(1, 1000), "kg": F(1),
     "lb": F(45359237, 100000000), "min": F(60), "h": F(3600), "day": F(86400), "week": F(604800), "s": F(1)}
CONV = [("ft", "yd"), ("in", "ft"), ("min", "h"), ("h", "day"), ("day", "week"), ("g", "lb"), ("kg", "lb"), ("m", "mile"), ("cm", "in"), ("m", "ft"), ("s", "min"), ("yd", "m")]
def dec(x, p):  # correct rounding half away from zero, as a decimal string with p decimals
    q = Decimal(x.numerator) / Decimal(x.denominator) if False else (Decimal(x.numerator) / Decimal(x.denominator))
    from decimal import getcontext
    getcontext().prec = 60
    q = Decimal(x.numerator) / Decimal(x.denominator)
    return str(q.quantize(Decimal(1).scaleb(-p), rounding=ROUND_HALF_UP))
def shift(s, p, d):  # last digit +-d
    n = int(s.replace(".", "")) + d; neg = n < 0; n = abs(n); t = str(n).rjust(p + 1, "0"); out = t[:-p] + "." + t[-p:]
    return ("-" if neg else "") + out
def trunc(x, p):
    n = int(abs(x) * 10 ** p); t = str(n).rjust(p + 1, "0"); return ("-" if x < 0 else "") + t[:-p] + "." + t[-p:]
def specs(w, r):
    out = []
    for _ in range(24):
        if r.random() < 0.5:
            if w == "A":
                a, b = r.randint(1, 300), r.choice([3, 6, 7, 9, 11, 13, 14, 17, 19, 21])
                if a % b == 0: a += 1
                out.append(("calc", f"{a}/{b}", F(a, b)))
            else:
                a, b, c, d = r.randint(1, 60), r.randint(1, 60), r.randint(2, 15), r.choice([7, 9, 11, 13, 17, 19])
                v = F((a + b) * c, d)
                out.append(("calc", f"({a}+{b})*{c}/{d}", v))
        else:
            fr, to = r.choice(CONV); q = r.randint(2, 900) if w == "A" else r.randint(2, 99) + r.choice([0, 0.5, 0.25]) if False else r.randint(2, 900)
            val = F(q) * U[fr] / U[to]
            out.append(("units", f"conv {q} {fr} -> {to}", val))
    return out
def main():
    items = {}
    for w in "AB":
        for s in range(1, 91):
            r = random.Random(s * 7243 + (1 if w == "B" else 0)); rows = []
            for tool, spec, v in specs(w, r):
                rows.append(dict(tool=tool, spec=spec, kind="exact", claim=str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}", want=("accept")))
                if v.denominator != 1 and (v * 10 ** 3).denominator != 1:   # non-terminating or long: rounding claims are meaningful
                    for p in (2, 3):
                        c = dec(v, p)
                        if F(c) == v: continue
                        rows.append(dict(tool=tool, spec=spec, kind=f"round{p}", claim=c, want="accept"))
                        if p == 2:
                            rows.append(dict(tool=tool, spec=spec, kind="near", claim=shift(c, p, 1), want="reject"))
                            rows.append(dict(tool=tool, spec=spec, kind="near", claim=shift(c, p, -1), want="reject"))
                            t = trunc(v, p)
                            if t != c: rows.append(dict(tool=tool, spec=spec, kind="trunc", claim=t, want="reject"))
                    if dec(v * F(3, 2), 2) != dec(v, 2): rows.append(dict(tool=tool, spec=spec, kind="gross", claim=dec(v * F(3, 2), 2), want="reject"))
            items[f"{w}{s}"] = rows
    json.dump(items, open(os.path.join(ROOT, "results", "n4_rounding_items.json"), "w"))
    print(sum(len(v) for v in items.values()), {k: sum(1 for v in items.values() for r in v if r["kind"] == k) for k in ("exact", "round2", "round3", "near", "trunc", "gross")})
if __name__ == "__main__": main()
