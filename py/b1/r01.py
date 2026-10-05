from rounds import *
print("R1 alpha sweep on TUNE (domain axes)")
res = {}
for a in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0):
    t = cand(TUNE, alpha=a); res[a] = {ax: tune_mean(t, ax) for ax in ("domain_r1", "domain_mrr", "domain_route")}
    print(a, {k: round(v, 4) for k, v in res[a].items()})
best = max(res, key=lambda a: res[a]["domain_mrr"]); print("best alpha on TUNE by domain_mrr:", best)
c = cand(alpha=best); b = base_tab()
compare(b, c, ["domain_r1", "domain_mrr", "domain_route"])
