"""R52 lookup-claim items (facts from data_verbal: CAPITALS, AUTHORS, ELEMENTS). Per (world, seed) 20 facts/table.
kinds: exact | case | ws | period | quote | article (accept for name tables) ; typo | wrong | qualifier | prefix (reject) ; elements: exact accept, case_flip/wrong reject."""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_verbal as V
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
def tables():
    cap = [(c, cap_.title()) for c, cap_ in V.CAPITALS]; au = [(b, a) for a, b in V.AUTHORS]; el = [(e, s) for e, s in V.ELEMENTS]
    return {"capital": cap, "author": au, "element": el}
def typo(v, r):
    if len(v) < 4: return v + "x"
    i = r.randrange(1, len(v) - 2); return v[:i] + v[i + 1] + v[i] + v[i + 2:] if v[i] != v[i + 1] else v[:i] + v[i + 2:]
def main():
    T = tables(); out = {}
    for w in "AB":
        for s in range(1, 91):
            r = random.Random(s * 5501 + (1 if w == "B" else 0)); rows = []
            for tb, facts in T.items():
                for key, val in r.sample(facts, min(20, len(facts))):
                    spec = f"{tb}.{key}"; add = lambda kind, claim, want: rows.append(dict(table=tb, key=key, value=val, spec=spec, kind=kind, claim=claim, want=want))
                    add("exact", val, "accept")
                    others = [v for k, v in facts if v != val]
                    if tb == "element":
                        flip = val.swapcase(); 
                        if flip != val: add("case_flip", flip, "reject")
                        add("wrong", r.choice(others), "reject")
                        continue
                    add("case", val.lower() if r.random() < .5 else val.upper(), "accept")
                    add("ws", "  " + val.replace(" ", "  ") + " ", "accept" if " " not in val else "reject_ws_inner")
                    add("period", val + ".", "accept"); add("quote", '"' + val + '"', "accept"); add("article", "the " + val.lower(), "accept")
                    t = typo(val, r)
                    if t != val: add("typo", t, "reject")
                    add("wrong", r.choice(others), "reject"); add("qualifier", val + ", Texas", "reject"); add("prefix", val[:-1], "reject")
            # drop rows with ambiguous ground truth (inner double-space on multiword values)
            out[f"{w}{s}"] = [x for x in rows if x["want"] in ("accept", "reject")]
    json.dump(out, open(os.path.join(ROOT, "results", "n5_lookup_items.json"), "w"))
    print(sum(len(v) for v in out.values()))
if __name__ == "__main__": main()
