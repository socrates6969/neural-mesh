"""k4 summary: consolidates results/k4_{taxo,links,links_qe,vers,extend}.json into Holm-adjusted primary tables (families pre-declared below) + TOST counts.
Usage: python py/k4_summary.py > results/k4_summary.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import holm, RES


def J(n):
    return json.load(open(os.path.join(RES, n)))


def fam(title, rows):
    """rows: (label, d, lo, hi, p) -> print Holm-adjusted table"""
    ps = [r[4] for r in rows]
    adj = holm(ps) if ps else []
    print("\n## " + title + "   (Holm over %d tests)" % len(rows))
    for r, a in zip(rows, adj):
        print("  %-62s d %+.4f [%+.4f,%+.4f]  p %.3g  holm %.3g %s" % (r[0], r[1], r[2], r[3], r[4], a, "*" if a < 0.05 else ""))


T = J("k4_taxo.json")
print("# K4 SUMMARY (real text/structure; outcome/demand models SYNTHETIC where noted)")
print("taxonomy picks: tune pick %s; by kind %s" % (T["pick"], T["pick_by"]))
rows = []
tost = []
for key, blk in T["phases"].items():
    for n, cc in blk["contrasts"].items():
        c = cc["mrr"]
        rows.append(("%s  %s  MRR" % (key, n), c["d"], c["lo"], c["hi"], c["p"]))
        tost.append((key, n, c["tost_pm01"], cc["hit8"]["tost_pm02"], cc["cost"]))
fam("TAXONOMY two-stage vs flat hybrid+spell, MRR@10 (seed-paired t)", rows)
print("  TOST +-.01 on MRR true in %d/%d; TOST +-.02 on hit@8 true in %d/%d; mean scan cost %.3f" % (sum(t[2] for t in tost), len(tost), sum(t[3] for t in tost), len(tost), np.mean([t[4] for t in tost])))
rows = []
for key, blk in T["phases"].items():
    for n, v in blk.get("e2e_dq_vs_flat_SYNTHETIC", {}).items():
        rows.append(("%s  %s  e2e dq (SYNTHETIC outcome model)" % (key, n), v["d"], v["lo"], v["hi"], v["p"]))
fam("TAXONOMY e2e quality under the h20 harm model (SYNTHETIC; auto clusters)", rows)

L = J("k4_links.json")
print("\nlinks: r frozen = %s ; tune gains by r %s" % (L["r_frozen"], L["tune_r_gain"]))
rows = []; rows_c = []
for ph, blk in L["results"].items():
    if not isinstance(blk, dict) or ph.endswith("r_sweep_union_nc"):
        continue
    for k, v in blk.items():
        t, arm = k.split("|", 1)
        if v["both"] is None:
            continue
        c = v["both"]
        if arm.startswith("LE|union_nc"):
            rows.append(("%s %s %s both@8" % (ph, t, arm), c["d"], c["lo"], c["hi"], c["p"]))
        own = {"ADJ": "LE|adj", "SECNEXT": "LE|adj", "DEP": "LE|dep", "XREF": "LE|xref"}.get(t)
        if own and arm.startswith(own + "|"):
            rows_c.append(("%s %s %s both@8 (CIRCULAR: graph type = label type)" % (ph, t, arm), c["d"], c["lo"], c["hi"], c["p"]))
fam("LINKS non-circular expansion (union of every graph type EXCEPT the one that defined the pair) vs flat8, both-hop recall", rows)
fam("LINKS circular ceiling-type contrasts (graph == label structure)", rows_c)
sh = [(ph, k, v["both"]) for ph, blk in L["results"].items() if isinstance(blk, dict) and not ph.endswith("r_sweep_union_nc") for k, v in blk.items() if "|LE|shuf" in k.replace("|LE|", "|LE|") and v["both"] is not None and "shuf" in k]
print("  random-link control: %d cells, mean d both@8 %+.4f, TOST+-.01 in %d" % (len(sh), np.mean([x[2]["d"] for x in sh]), sum(x[2]["tost01"] for x in sh)))
print("  single-hop control:", {k: (round(v["d"], 4), v["tost01"], v["tost02"]) for k, v in L["control_singlehop"].items()})
if os.path.exists(os.path.join(RES, "k4_links_qe.json")):
    Q = J("k4_links_qe.json")
    rows = []
    for k, c in Q["results"].items():
        if k.endswith("union_nc"):
            rows.append(("QE " + k, c["d"], c["lo"], c["hi"], c["p"]))
    fam("LINKS graph-based query expansion (beta=%s) vs flat8" % Q["beta"], rows)

V = J("k4_vers.json")
print("\n## VERSIONS (tau %s; n queries %s; crate-cluster bootstrap, Holm inside k4_vers.txt)" % (V["tau"], V["n_queries"]))
for k, b in V["results"].items():
    if k.split("|")[1] in ("changed", "STABLE", "ONLYOLD"):
        print("  %-18s n=%-4d correct@1 flat %.3f key %.3f keylist %.3f sem %.3f | sem-flat %+.3f [%+.3f,%+.3f]  keylist-flat %+.3f [%+.3f,%+.3f]" % (k, b["n"], b["mean"]["flat"], b["mean"]["key"], b["mean"]["keylist"], b["mean"]["sem"], b["contrast"]["sem"]["d"], b["contrast"]["sem"]["lo"], b["contrast"]["sem"]["hi"], b["contrast"]["keylist"]["d"], b["contrast"]["keylist"]["lo"], b["contrast"]["keylist"]["hi"]))
print("  signal:", json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in V["signal"].items()}))
print("  key reliability:", V["key_reliability"])

E = J("k4_extend.json")
print("\nextend: frozen %s" % E["params"])
for arm in ("gap", "topic_pred"):
    for met in ("cov_auc", "rec_auc"):
        rows = []
        for k, b in E["results"].items():
            if k.endswith("|hot"):
                c = b["contrast"]["%s|%s" % (arm, met)]
                rows.append(("%s %s-random %s" % (k, arm, met), c["d"], c["lo"], c["hi"], c["p"]))
        fam("ACQUISITION %s vs random (hot demand), %s growth-curve mean over budgets" % (arm, met), rows)
for mode in ("uniform", "drift"):
    for arm in ("gap", "topic_pred"):
        print("\n## control %s demand, %s - random cov_auc | rec_auc:" % (mode, arm))
        for k, b in E["results"].items():
            if k.endswith("|" + mode):
                c = b["contrast"]["%s|cov_auc" % arm]; r = b["contrast"]["%s|rec_auc" % arm]
                print("  %-22s cov %+.4f [%+.4f,%+.4f] TOST.01=%s TOST.02=%s | rec %+.4f [%+.4f,%+.4f]" % (k, c["d"], c["lo"], c["hi"], c["tost01"], c["tost02"], r["d"], r["lo"], r["hi"]))
print("\n## ingestion-order baselines vs random (cov_auc, hot): recent / largest")
for k, b in E["results"].items():
    if k.endswith("|hot"):
        print("  %-22s recent %+.4f [%+.4f,%+.4f] largest %+.4f oracle-topic %+.4f oracle-unit %+.4f gap %+.4f topic_pred %+.4f" % (k, b["contrast"]["recent|cov_auc"]["d"], b["contrast"]["recent|cov_auc"]["lo"], b["contrast"]["recent|cov_auc"]["hi"], b["contrast"]["largest|cov_auc"]["d"], b["contrast"]["topic_oracle|cov_auc"]["d"], b["contrast"]["oracle|cov_auc"]["d"], b["contrast"]["gap|cov_auc"]["d"], b["contrast"]["topic_pred|cov_auc"]["d"]))
print("\n## gap-detector AUC (label-free low-cosine signal vs 'relevant unit not ingested'):", {k: round(b["meta"]["detect_auc"], 3) for k, b in E["results"].items() if k.endswith("|hot")})
