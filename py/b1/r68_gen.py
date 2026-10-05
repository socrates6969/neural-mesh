"""R68: fresh randomly generated identity claims (expression-tree rewriting), independent of any probe list.
true  = E vs E' where E' is E after random EXACT rewrites (verified with python Fractions on random rationals: 0 failures required)
mut   = E vs a MUTATION of E'. Ground truth 'differs' verified at random rationals (a mutant that is secretly equal is dropped).
kind 'near' = mutant that agrees with E on every integer of -12..13 (so the integer-grid verifier cannot see it); kind 'diff' = visible on the grid.
Seeds 91-150 (fresh, never used before), worlds A (x) and B (x,y)."""
import random, json, os, itertools
from fractions import Fraction as F
SD0, SD1 = (91, 151) if len(__import__("sys").argv) < 3 else (int(__import__("sys").argv[1]), int(__import__("sys").argv[2]))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
def ev(t, env):
    k = t[0]
    if k == "n": return F(t[1])
    if k == "v": return env[t[1]]
    if k == "+": return ev(t[1], env) + ev(t[2], env)
    if k == "-": return ev(t[1], env) - ev(t[2], env)
    if k == "*": return ev(t[1], env) * ev(t[2], env)
    if k == "^2": return ev(t[1], env) ** 2
    if k == "abs": return abs(ev(t[1], env))
    import math
    if k == "floor": return F(math.floor(ev(t[1], env)))
    if k == "ceil": return F(math.ceil(ev(t[1], env)))
    if k == "min": return min(ev(t[1], env), ev(t[2], env))
    if k == "max": return max(ev(t[1], env), ev(t[2], env))
    raise KeyError(k)
def s(t):
    k = t[0]
    if k == "n": return str(t[1]) if t[1] >= 0 else f"(0-{-t[1]})"
    if k == "v": return t[1]
    if k in "+-*": return f"({s(t[1])}{k}{s(t[2])})"
    if k == "^2": return f"({s(t[1])})^2"
    return f"{k}({s(t[1])})" if k in ("abs", "floor", "ceil") else f"{k}({s(t[1])},{s(t[2])})"
def rand_tree(r, vs, d):
    if d == 0 or r.random() < .2: return ("v", r.choice(vs)) if r.random() < .75 else ("n", r.randint(1, 9))
    k = r.choice(["+", "-", "*", "^2", "abs", "floor", "ceil", "min", "max", "+", "*"])
    if k in ("^2", "abs", "floor", "ceil"): return (k, rand_tree(r, vs, d - 1))
    return (k, rand_tree(r, vs, d - 1), rand_tree(r, vs, d - 1))
def nodes(t, path=()):
    yield path, t
    for i, c in enumerate(t[1:], 1):
        if isinstance(c, tuple): yield from nodes(c, path + (i,))
def get(t, p):
    for i in p: t = t[i]
    return t
def put(t, p, v):
    if not p: return v
    l = list(t); l[p[0]] = put(t[p[0]], p[1:], v); return tuple(l)
def neg(a): return ("-", ("n", 0), a)
def exact_rewrites(t, r):
    k = t[0]; out = []
    if k in ("+", "*"): out.append((k, t[2], t[1]))
    if k == "-": out.append(("+", t[1], neg(t[2])))
    if k == "*" and t[2][0] == "+": out.append(("+", ("*", t[1], t[2][1]), ("*", t[1], t[2][2])))
    if k == "^2" and t[1][0] == "+": a, b = t[1][1], t[1][2]; out.append(("+", ("+", ("^2", a), ("*", ("n", 2), ("*", a, b))), ("^2", b)))
    if k == "^2": out.append(("*", t[1], t[1]))
    if k == "min": out.append(neg(("max", neg(t[1]), neg(t[2]))))
    if k == "max": out.append(neg(("min", neg(t[1]), neg(t[2]))))
    if k == "ceil": out.append(neg(("floor", neg(t[1]))))
    if k == "floor": out.append(neg(("ceil", neg(t[1]))))
    if k == "abs" and t[1][0] == "*": out.append(("*", ("abs", t[1][1]), ("abs", t[1][2])))
    if k == "abs": out.append(("max", t[1], neg(t[1])))
    if k == "floor" and t[1][0] == "+" and t[1][2][0] == "n": out.append(("+", ("floor", t[1][1]), t[1][2]))
    if k == "+": out.append(("-", t[1], neg(t[2])))
    return out
def mutate(t, r, vs):
    nd = list(nodes(t)); fam = r.choice(["generic", "generic", "frac", "wrap"]); p, sub = r.choice(nd)
    if fam == "frac":  # add a term that vanishes on every integer
        v = ("v", r.choice(vs)); return ("+", t, r.choice([("-", v, ("floor", v)), ("-", ("ceil", v), v)]))
    if fam == "wrap":  # wrap a subterm in an integer-grid-neutral clamp / rounding
        big = r.choice([40, 99, 5000000]); w = r.choice([("floor", sub), ("ceil", sub), ("min", sub, ("n", big)), ("max", sub, ("n", -big))]); return put(t, p, w)
    k = sub[0]
    if k == "n": return put(t, p, ("n", sub[1] + r.choice([-1, 1, 2])))
    if k in "+-*" and len(k) == 1: return put(t, p, ({"+": "-", "-": "+", "*": "+"}[k], sub[1], sub[2]))
    if k == "min": return put(t, p, ("max", sub[1], sub[2]))
    if k == "max": return put(t, p, ("min", sub[1], sub[2]))
    if k == "floor": return put(t, p, ("ceil", sub[1]))
    if k == "ceil": return put(t, p, ("floor", sub[1]))
    if k == "abs": return put(t, p, sub[1])
    if k == "^2": return put(t, p, ("*", ("n", 2), sub[1]))
    if k == "v": return put(t, p, ("+", sub, ("n", 1)))
    return t
def rnd_env(r, vs): return {v: F(r.randint(-400, 400), r.choice([1, 2, 3, 7, 13, 101, 997])) for v in vs}
def grid_equal(a, b, vs):
    pts = range(-12, 14)
    for g in (itertools.product(pts, repeat=len(vs))):
        env = {v: F(x) for v, x in zip(vs, g)}
        if ev(a, env) != ev(b, env): return False
    return True
def gen_world(w, seed):
    r = random.Random(seed * 90007 + (w == "B")); vs = ["x"] if w == "A" else ["x", "y"]; out = {"true": [], "near": [], "diff": []}
    tries = 0
    while (len(out["true"]) < 12 or len(out["near"]) < 8 or len(out["diff"]) < 12) and tries < 4000:
        tries += 1; E = rand_tree(r, vs, 3); P = E
        for _ in range(r.randint(1, 3)):
            c = list(nodes(P)); r.shuffle(c)
            for p, sub in c:
                rw = exact_rewrites(sub, r)
                if rw: P = put(P, p, r.choice(rw)); break
        if P == E: continue
        envs = [rnd_env(r, vs) for _ in range(40)]
        if any(ev(E, e) != ev(P, e) for e in envs): raise SystemExit("exact rewrite broken: " + s(E) + " vs " + s(P))
        if len(out["true"]) < 12 and len(s(E)) < 160 and len(s(P)) < 160: out["true"].append(f"{s(E)} == {s(P)}")
        M = mutate(P, r, vs)
        if all(ev(E, e) == ev(M, e) for e in envs): continue
        if len(s(M)) > 170: continue
        g = grid_equal(E, M, vs) if len(vs) == 1 or len(s(M)) < 120 else None
        if g is None: continue
        k = "near" if g else "diff"
        if len(out[k]) < (8 if k == "near" else 12): out[k].append(f"{s(E)} == {s(M)}")
    return out
if __name__ == "__main__":
    res = {}; tot = {"true": 0, "near": 0, "diff": 0}
    for w in "AB":
        for sd in range(SD0, SD1):
            o = gen_world(w, sd); res[f"{w}{sd}"] = [dict(kind=k, spec=x) for k, L in o.items() for x in L]
            for k in tot: tot[k] += len(o[k])
    json.dump(res, open(os.path.join(ROOT, "results", ("r68_identity_items.json" if SD0 == 91 else f"r68_identity_items_{SD0}.json")), "w")); print(tot)
