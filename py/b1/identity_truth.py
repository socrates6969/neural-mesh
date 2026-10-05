"""Independent ground-truth check (python Fractions) of the generated identity items: true => equal on all probes; near => equal on integer grid -12..13 but differ somewhere; diff => differs on grid."""
import json, re, os, itertools, math
from fractions import Fraction as F
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
def fl(x): return F(math.floor(x)); 
def ce(x): return F(math.ceil(x))
def compile_side(s):
    s = re.sub(r"(\d+)", r"F(\1)", s).replace("^", "**")
    return eval("lambda x,y: " + s, dict(F=F, floor=fl, ceil=ce, abs=abs, min=min, max=max))
def sides(spec): l, r = spec.split("=="); return compile_side(l), compile_side(r)
GRID = list(range(-12, 14))
PROBE = [F(1, 2), F(-1, 2), F(3, 2), F(-3, 2), F(1, 3), F(7, 3), F(-7, 3), F(22, 7), F(100), F(-100), F(1000), F(-1000), F(1, 1000), F(10**6)]
EX1 = [(F(1, 5), 0), (F(5000001+1000), 0), (F(-6000000), 0), (F(1, 11), 0)]
EX2 = [(F(1, 5), F(1, 3)), (F(5001000), F(3)), (F(-6000000), F(1)), (F(1, 11), F(7)), (F(1, 5), F(1, 5)), (F(3, 5), F(9, 5)),(F(1,11),F(2,7))]
def agree(l, r, pts2):
    for x, y in pts2:
        try: a, b = l(x, y), r(x, y)
        except ZeroDivisionError: continue
        if a != b: return False
    return True
def check_all():
    items = json.load(open(os.path.join(ROOT, "results", "n1_identity_items.json"))); bad = 0; n = 0
    for k, its in items.items():
        two = k[0] == "B"
        for it in its:
            l, r = sides(it["spec"]); n += 1
            g = [(F(a), F(b)) for a in GRID for b in (GRID if two else [0])]
            p = [(a, b) for a in PROBE + GRID for b in (PROBE[:6] + [F(2), F(-5)] if two else [0])]
            ig = agree(l, r, g); ip = agree(l, r, g + p)
            ok = {"true": ig and ip, "near": ig and not ip, "diff": not ig, "evade": ig and ip and not agree(l, r, EX2 if two else EX1)}[it["kind"]]
            if not ok:
                bad += 1
                if bad <= 10: print("BAD", k, it, ig, ip)
    print("checked", n, "bad", bad)
if __name__ == "__main__": check_all()
