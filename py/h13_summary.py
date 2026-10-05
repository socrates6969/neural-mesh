"""h13: print the compact tables used in results/h13_board_entry.md from results/h13_retrieval.json, h13_route.json, h13_corpus_stats.json.
Usage: python py/h13_summary.py > results/h13_summary.txt"""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
R = json.load(open(os.path.join(RES, "h13_retrieval.json")))
Q = json.load(open(os.path.join(RES, "h13_route.json")))
C = json.load(open(os.path.join(RES, "h13_corpus_stats.json")))
SPL = [("held_A_31_60", "A held"), ("confirm_A_61_90", "A conf"), ("confirm_B_61_90", "B conf")]

print("=== RETRIEVAL levels mix25 hit1/MRR/F1@3 (and P@3, R@3 in brackets for hybz_s) ===")
for sys in ["dense_raw", "dense_s", "bm25_raw", "bm25_s", "hybz_raw", "hybz_s", "hybrrf_raw", "hybrrf_s", "hybrsf_raw", "hybrsf_s", "rerank_raw", "rerank_s"]:
    cells = []
    for k, _ in SPL:
        c = R[k]["levels"][sys]["mix25"]
        cells.append("%.3f/%.3f/%.3f" % (c["Hit1"], c["Mrr"], c["F1at3"]))
    print("%-11s %s" % (sys, " | ".join(cells)))
print("P@3 / R@3 hybz_s vs dense_s (mix25):", {k: [(round(R[k]["levels"][s]["mix25"]["P3"], 3), round(R[k]["levels"][s]["mix25"]["R3"], 3)) for s in ("dense_s", "hybz_s")] for k, _ in SPL})
print("\n=== RETRIEVAL contrasts, mix25 MRR (seed t-CI | cluster-bootstrap CI | holm_boot) ===")
for a, b in [("hybz_s", "dense_s"), ("hybrrf_s", "dense_s"), ("hybrsf_s", "dense_s"), ("rerank_s", "dense_s"), ("bm25_s", "dense_s"), ("hybz_s", "hybrrf_s"), ("hybz_s", "hybrsf_s"), ("hybz_s", "rerank_s"), ("dense_s", "dense_raw"), ("hybz_s", "hybz_raw"), ("dense_raw", "bm25_raw")]:
    for metric in ("Mrr", "Hit1", "F1at3"):
        cells = []
        for k, _ in SPL:
            r = [x for x in R[k]["compare"] if x["a"] == a and x["b"] == b and x["slice"] == "mix25" and x["metric"] == metric][0]
            cells.append("%+.4f t[%+.3f,%+.3f] b[%+.3f,%+.3f] h%.3f%s" % (r["diff_seed"], r["t_lo"], r["t_hi"], r["boot_lo"], r["boot_hi"], r["holm_boot"], {1: "*", -1: "!", 0: ""}[r["robust"]]))
        print("%-9s-%-9s %-5s " % (a, b, metric) + " | ".join(cells))
print("\nrobust wins/losses per split:", {k: (sum(1 for x in R[k]["compare"] if x["robust"] == 1), sum(1 for x in R[k]["compare"] if x["robust"] == -1), len(R[k]["compare"])) for k, _ in SPL})
print("\n=== overlap tertiles (hybz_s - dense_s MRR) ===")
for k, v in R["overlap_tertiles"].items():
    print(k, {t: (v[t]["overlap_range"], round(v[t]["hybz_minus_dense"]["d"], 4)) for t in ("t1", "t2", "t3")}, "corr", round(v["corr_overlap_vs_hybrid_gain"], 3))
print("\n=== overlap real ===")
for k, v in R["overlap_real"].items():
    print(k, {f: (round(v[f]["idf_cov"]["mean"], 3), round(v[f]["idf_cov"]["median"], 3)) for f in v})
print("=== overlap g1 ===")
for t in "AB":
    print("g1", t, {k: (round(v["idf_cov"]["mean"], 3), round(v["idf_cov"]["median"], 3)) for k, v in R["g1_bridge"][t]["overlap"].items()})
print("\n=== g1 bridge levels all hit1/MRR/F1@3 and contrasts MRR ===")
for t in "AB":
    g = R["g1_bridge"][t]
    for sys in ("dense_raw", "dense_s", "bm25_s", "hybz_s", "hybrrf_s", "hybrsf_s", "rerank_s"):
        c = g["levels"][sys]["all"]
        print("g1", t, "%-9s %.3f/%.3f/%.3f" % (sys, c["Hit1"], c["Mrr"], c["F1at3"]))
    for r in g["compare_all"]:
        if r["metric"] == "Mrr" and (r["a"], r["b"]) in [("hybz_s", "dense_s"), ("hybrrf_s", "dense_s"), ("hybrsf_s", "dense_s"), ("rerank_s", "dense_s"), ("bm25_s", "dense_s"), ("dense_s", "dense_raw")]:
            print("   g1", t, "%-9s-%-9s %+.4f [%+.4f,%+.4f] holm %.3f" % (r["a"], r["b"], r["diff"], r["lo"], r["hi"], r["holm"]))
print("\n=== corpus stats ===")
for k, v in C.items():
    print(k, {x: v[x] for x in ("n_docs", "mean_tokens", "vocab", "hapax_share_of_vocab", "nn_cos", "share_docs_with_jaccard_ge_.6_partner", "siblings_ge_.85_of_relevant_doc", "query_to_relevant_cos", "best_relevant_minus_best_nonrelevant_cos")})
print("\n=== latency / bytes ===")
print(json.dumps(R["latency_bytes"]["us_per_query"]), json.dumps(R["latency_bytes"]["bytes"]))

print("\n=== ROUTING (H2000) ===")
worlds = [w["name"] for w in Q["worlds"]]
print("tuned picks:", [(p["arm"], p["thr"], p["min_n"], p["k"]) for p in Q["picks"] if p["arm"].split("|")[1] in ("dense_s", "hybz_s", "rerank_s", "hybrsf_s", "oracle")])
for k, lab in SPL:
    k = {"held_A_31_60": "A_held_31_60", "confirm_A_61_90": "A_confirm_61_90", "confirm_B_61_90": "B_confirm_61_90"}[k]
    L = Q[k]["levels"]
    print("\n--", lab)
    print("%-16s" % "arm" + "".join("%11s" % w for w in worlds))
    best = {w: max(L[b][w]["H2000"] for b in ("keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded")) for w in worlds}
    print("%-16s" % "best fair (6)" + "".join("%11.4f" % best[w] for w in worlds))
    best7 = {w: max(best[w], L["global-ts"][w]["H2000"]) for w in worlds}
    print("%-16s" % "best fair (7)" + "".join("%11.4f" % best7[w] for w in worlds))
    for arm in ["global-ts", "guarded", "safe|dense_s", "safe|hybz_s", "safe|rerank_s", "safe|oracle", "g1|dense_s", "g1|hybz_s", "g1|rerank_s", "g1|oracle", "ctx|dense_s", "ctx|hybz_s", "ctx|oracle"]:
        print("%-16s" % arm + "".join("%11.4f" % L[arm][w]["H2000"] for w in worlds))
    # diagnostics of safe|dense_s
    for arm in ["safe|dense_s", "safe|hybz_s", "g1|dense_s", "g1|hybz_s", "ctx|dense_s", "ctx|hybz_s"]:
        print("  diag %-12s" % arm + " ".join("%s: vote %.2f prec %.2f anch %.2f" % (w, L[arm][w]["vote_frac"] or 0, L[arm][w]["inj_prec"] if L[arm][w]["inj_prec"] is not None else float("nan"), L[arm][w]["anchor_acc"] if L[arm][w]["anchor_acc"] is not None else float("nan")) for w in ("T12", "Tctx", "Tqa")))
    for fam, lab2 in (("vs_same_front", "vs same front"), ("vs_best_fair6", "vs best fair (6 pre-declared)"), ("vs_best_fair", "vs best fair (7 incl. global-ts)"), ("vs_safe_dense_s", "vs same-gate dense_s")):
        print(" ", lab2)
        for arm in ["safe|dense_s", "safe|hybz_s", "safe|rerank_s", "g1|dense_s", "g1|hybz_s", "g1|hybrsf_s", "g1|rerank_s", "ctx|dense_s", "ctx|hybz_s"]:
            row = []
            for w in worlds:
                r = [x for x in Q[k][fam] if x["arm"] == arm and x["world"] == w and x["metric"] == "H2000"]
                if not r:
                    row.append("%11s" % "-")
                    continue
                r = r[0]
                row.append("%+7.3f%s%s " % (r["diff"], "*" if r.get("win") else ("!" if r.get("loss") else " "), "e" if r.get("tost_holm_pm01") is not None and r["tost_holm_pm01"] < 0.05 else " "))
            print("    %-14s" % arm + "".join("%11s" % x for x in row))
    print("  verdicts (keep_vs_front / keep_vs_bestfair / beats_dense_s):", {a: (v["keep_vs_front"], v["keep_vs_bestfair"], v["beats_safe_dense_s"]) for a, v in Q["verdicts"][k].items() if a.split("|")[1] in ("dense_s", "hybz_s", "hybrrf_s", "hybrsf_s", "rerank_s", "bm25_s", "oracle")})
print("\n=== POST-HOC Tqa ctx sweep pairs (thr .30) ===")
for lab, v in Q["posthoc_ctx_sweep"].items():
    for r in v["Tqa_pairs"]:
        if r["thr"] == "0.30":
            print(lab, "%-9s-%-9s %+.4f [%+.4f,%+.4f] holm %.3f" % (r["a"], r["b"], r["diff"], r["lo"], r["hi"], r["holm"]))
print("\n=== loop us/task (loaded box) A held T12 ===")
L = Q["A_held_31_60"]["levels"]
for arm in ["guarded", "safe|dense_s", "safe|hybz_s", "g1|dense_s", "ctx|dense_s", "safe|oracle"]:
    print(arm, round(L[arm]["T12"]["us_per_task"], 1))
