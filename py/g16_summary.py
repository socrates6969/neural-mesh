import json,sys
f=sys.argv[1] if len(sys.argv)>1 else "results/g16_route.json"
d=json.load(open(f))
rows=d["comparisons"]
fam=sys.argv[2] if len(sys.argv)>2 else "F6:vs-same-front(guarded)"
arms=sys.argv[3].split("|") if len(sys.argv)>3 else ["R0 (g1 retrieval)","G-post+g1","G-shrink+g1","G-shrink+g1(inj-gated)","G-shrink+g1+leninj","FINAL[mini-hybrid]"]
worlds=[]
for r in rows:
    if r["fam"]==fam and r["world"] not in worlds: worlds.append(r["world"])
print("==",fam,f)
for w in worlds:
    print("--",w)
    for a in arms:
        line=f"   {a:<26}"
        for m in ("H2000","late"):
            for r in rows:
                if r["fam"]==fam and r["world"]==w and r["arm"]==a and r["metric"]==m:
                    line+=f" | {m} {r['diff']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']}"
        print(line)
