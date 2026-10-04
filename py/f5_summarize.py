import json,sys
def load(p): return json.load(open(p))
for tag,kf,sf in [("31-60","results/f5_knowledge.json","results/f5_shaper.json"),("61-90","results/f5_knowledge_confirm61-90.json","results/f5_shaper_confirm61-90.json")]:
    k=load(kf)
    print("=====",tag)
    for src in ["mini","hash","synth"]:
        for reg in ["W","O"]:
            for kind in ["regime","ablation"]:
                d=k[f"{src}_{'regime' if kind=='regime' else 'ablation'}_{reg}"]
                print(f"[{src} {kind} {reg}] holm W/L {d['holm_wins']}/{d['holm_losses']}")
                for c in d["cells"]:
                    if c["H"]==2000 and c["candidate"] in("kn-feat(retuned)","kn-d+vote(retuned)","kn-feat(d6cfg)","kn-d+vote(d6cfg)"):
                        pass
                rows={}
                for c in d["cells"]:
                    if c["H"]==2000: rows.setdefault(c["world"],{})[c["candidate"]]=c
                for w,r in rows.items():
                    s="  %-22s"%w
                    for cn in ["kn-feat(d6cfg)","kn-feat(retuned)","kn-d+vote(d6cfg)","kn-d+vote(retuned)"]:
                        c=r[cn]; m="*" if c["holm_p"]<.05 and c["diff"]>0 else ("!" if c["holm_p"]<.05 else " ")
                        s+=" %+.3f[%+.3f,%+.3f]%s"%(c["diff"],c["ci"][0],c["ci"][1],m)
                    print(s)
    s=load(sf)
    for key in ["mini_main","mini_ablation","hash_main","hash_ablation"]:
        d=s["doc"][key]; print(f"[shaper {key}] holm W/L {d['holm_wins']}/{d['holm_losses']}")
        rows={}
        for c in d["cells"]:
            if c["H"]==2000: rows.setdefault(c["world"],{})[c["candidate"]]=c
        for w,r in rows.items():
            s2="  %-24s"%w
            for cn,c in r.items():
                m="*" if c["holm_p"]<.05 and c["diff"]>0 else ("!" if c["holm_p"]<.05 else " ")
                s2+=" %+.3f[%+.3f,%+.3f]%s"%(c["diff"],c["ci"][0],c["ci"][1],m)
            print(s2)
