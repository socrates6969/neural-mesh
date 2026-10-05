"""R39 identity/near-miss false-ACCEPT axis. Per (world, seed): 12 items per kind.
kinds: true (genuine identities), near (NOT identical but agree on every integer point of the verifier's -12..13 grid), diff (plainly different).
World A = one variable, world B = two variables. Items are authored by templates (synthetic)."""
import json, random, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
def gen_A(r):
    a, b, c = r.randint(2, 9), r.randint(2, 9), r.randint(1, 7)
    T = r.randint(14, 60); M = r.randint(2, 6)
    true = [f"(x+{a})^2 == x^2+{2*a}*x+{a*a}", f"(x+{a})*(x-{a}) == x^2-{a*a}", f"x^3-{a**3} == (x-{a})*(x^2+{a}*x+{a*a})",
            f"max(x,{a})+min(x,{a}) == x+{a}", f"abs({b}*x) == {b}*abs(x)", f"floor(x) == 0-ceil(0-x)", f"{a}*(x+{b}) == {a}*x+{a*b}",
            f"floor(x+{c}) == floor(x)+{c}", f"ceil(x)+{c} == ceil(x+{c})", f"min(x,{T}) == 0-max(0-x,0-{T})", f"(x+{a})^3 == x^3+{3*a}*x^2+{3*a*a}*x+{a**3}", f"x*(x+{a}) == x^2+{a}*x"]
    near = [f"floor(x) == x", f"ceil(x) == x", f"floor(x+{c}) == x+{c}", f"ceil({M}*x) == {M}*x", f"min(x,{T}) == x", f"max(x,0-{T}) == x",
            f"min(x*x,{T*T}) == x*x", f"max(x*x*x,0-{T**3}) == x*x*x", f"floor({M}*x) == {M}*x", f"floor(x)+ceil(x) == 2*x",
            f"min(abs(x),{T}) == abs(x)", f"ceil(x+{c}) == floor(x)+{c}"]
    diff = [f"(x+{a})^2 == x^2+{a*a}", f"(x+{a})*(x-{a}) == x^2+{a*a}", f"x^3-{a**3} == (x-{a})*(x^2+{a*a})", f"max(x,{a}) == x", f"abs(x) == x",
            f"{a}*(x+{b}) == {a}*x+{b}", f"x*(x+{a}) == x^2+{a+1}*x", f"(x+{a})^3 == x^3+{a**3}", f"min(x,{a}) == x", f"x^2 == x", f"floor(x) == ceil(x)+1", f"x+{a} == x+{b+a+1}"]
    evade = [f"floor(42000*x) == 42000*x", f"min(x,{5000000+r.randint(0,999)}) == x", f"max(x,0-{5000000+r.randint(0,999)}) == x", f"ceil(84000*x) == 84000*x"]
    return true, near, diff, evade
def gen_B(r):
    a, b = r.randint(2, 9), r.randint(2, 9); T = r.randint(27, 70); M = r.randint(2, 5); U = r.randint(170, 300)
    true = [f"(x+y)^2 == x^2+2*x*y+y^2", f"(x-y)*(x+y) == x^2-y^2", f"max(x,y)+min(x,y) == x+y", f"abs(x-y) == abs(y-x)", f"min(x,y) == 0-max(0-x,0-y)",
            f"{a}*(x+y) == {a}*x+{a}*y", f"(x+{a}*y)^2 == x^2+{2*a}*x*y+{a*a}*y^2", f"floor(x)+floor(y) <= x+y".replace("<=", "=="),  # placeholder replaced below
            f"abs(x*y) == abs(x)*abs(y)", f"x*y+x*{b} == x*(y+{b})", f"max(x,y) == (x+y+abs(x-y))/2", f"(x+y)^3 == x^3+3*x^2*y+3*x*y^2+y^3"]
    true[7] = f"floor(x+{a}) == floor(x)+{a}"
    near = [f"floor(x*y) == x*y", f"min(x+y,{T}) == x+y", f"max(x*y,0-{U}) == x*y", f"ceil(x+y) == x+y", f"min(x*y,{U}) == x*y", f"floor(x/{M}*{M}) == x",
            f"ceil({M}*x+y) == {M}*x+y", f"max(x+y,0-{T}) == x+y", f"floor(x)+floor(y) == x+y", f"min(abs(x*y),{U}) == abs(x*y)", f"ceil(x)+ceil(y) == x+y", f"floor(x*{M}+y) == x*{M}+y"]
    diff = [f"(x+y)^2 == x^2+y^2", f"(x-y)^2 == x^2-y^2", f"max(x,y) == x", f"abs(x-y) == x-y", f"min(x,y) == x+y-max(x,y)+1", f"{a}*(x+y) == {a}*x+y",
            f"(x+{a}*y)^2 == x^2+{a*a}*y^2", f"floor(x+y) == floor(x)+floor(y)+1", f"abs(x*y) == x*y", f"x*y+x*{b} == x*y+{b}", f"max(x,y) == (x+y+abs(x-y))", f"(x+y)^3 == x^3+y^3"]
    evade = [f"floor(1764000000*x*y) == 1764000000*x*y", f"min(x*y,{5000000+r.randint(0,999)}) == x*y", f"ceil(42000*x+42000*y) == 42000*x+42000*y", f"max(x+y,0-{5000000+r.randint(0,999)}) == x+y"]
    return true, near, diff, evade
def main():
    out = {}
    for w, g in (("A", gen_A), ("B", gen_B)):
        for s in range(1, 91):
            r = random.Random(s * 4177 + (1 if w == "B" else 0)); t, n, d, e = g(r)
            out[f"{w}{s}"] = [dict(kind="true", spec=x) for x in t] + [dict(kind="near", spec=x) for x in n] + [dict(kind="diff", spec=x) for x in d] + [dict(kind="evade", spec=x) for x in e]
    json.dump(out, open(os.path.join(ROOT, "results", "n1_identity_items.json"), "w"))
    print(sum(len(v) for v in out.values()))
if __name__ == "__main__": main()
