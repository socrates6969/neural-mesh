#!/usr/bin/env python
"""Compact tables from results/h7_<exp>_<worlds>_<set>.json  (usage: python results/h7_summarize.py exp worlds ref "arm substring" [more arm substrings])"""
import json, sys
exp, worlds, ref = sys.argv[1:4]
subs = sys.argv[4:]
for st in ("held", "confirm"):
    try:
        d = json.load(open(f"results/h7_{exp}_{worlds}_{st}.json"))
    except Exception as e:
        print(st, "missing", e); continue
    ws = [m["world"] for m in d["means"]]
    print(f"== {exp} {worlds} {st}  diff vs [{ref}]  (creative metric; * Holm win, ! Holm loss; cost = extra gen+judge calls per creative task)")
    print("%-58s" % "arm" + "".join("%12.11s" % w for w in ws))
    arms = []
    for t in d["tests"]:
        if t["ref"].startswith(ref) and t["metric"] == "creative" and t["arm"] not in arms and any(s in t["arm"] for s in subs):
            arms.append(t["arm"])
    for a in arms:
        row = "%-58.58s" % a
        for w in ws:
            x = [t for t in d["tests"] if t["arm"] == a and t["world"] == w and t["metric"] == "creative" and t["ref"].startswith(ref)]
            if x:
                t = x[0]; m = "*" if t["t"]["diff"] > 0 and t["holm"] < .05 else ("!" if t["t"]["diff"] < 0 and t["holm"] < .05 else " ")
                row += "%+11.4f%s" % (t["t"]["diff"], m)
            else:
                row += "%12s" % "-"
        print(row)
