"""h17 summary tables from results/h17_*.json  (no new computation).  Usage: python py/h17_summary.py > results/h17_summary.txt"""
import json, os, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = lambda n: json.load(open(os.path.join(R, n), encoding="utf8"))
modes = ["mini", "hash", "hashtok"]
ret = {m: J("h17_retrieval_%s.json" % m) for m in modes}
SPL = [("held_A_31_60", "A held"), ("confirm_A_61_90", "A conf"), ("confirm_B_61_90", "B conf")]
f = lambda x: "%.3f" % x


def sect(t):
    print("\n" + "=" * 8, t, "=" * 8)


sect("1. REAL doc text: doc-level levels (mix25 = 75% clean + 25% typo queries). cell = hit@1 / MRR / F1@3 / P@3 / R@3")
for sp, sn in SPL:
    print("--", sn)
    for sysn in ["dense_raw", "dense_s", "bm25_raw", "bm25_s", "hybz_raw", "hybz_s"]:
        row = []
        for m in modes:
            c = ret[m][sp]["levels"][sysn]["mix25"]
            row.append("%s: %s/%s/%s/%s/%s" % (m, f(c["Hit1"]), f(c["Mrr"]), f(c["F1at3"]), f(c["P3"]), f(c["R3"])))
        print("  %-10s" % sysn, " | ".join(row))
sect("2. REAL doc text: contrasts, mix25 (diff, cluster-bootstrap 95% CI, robust = Holm over 126 under seed-t AND bootstrap; + win, - loss, . n.s.)")
pairs = [("hybz_s", "dense_s"), ("dense_s", "dense_raw"), ("bm25_s", "dense_s"), ("hybz_s", "bm25_s"), ("hybz_s", "hybz_raw"), ("hybz_raw", "dense_raw")]
for a, b in pairs:
    for met in ["Mrr", "Hit1", "F1at3"]:
        print(" %-8s - %-9s %-5s" % (a, b, met))
        for sp, sn in SPL:
            cells = []
            for m in modes:
                e = [x for x in ret[m][sp]["compare"] if x["a"] == a and x["b"] == b and x["slice"] == "mix25" and x["metric"] == met]
                if not e:
                    cells.append("%s: n/a" % m)
                    continue
                e = e[0]
                cells.append("%s: %+.3f [%+.3f,%+.3f] %s" % (m, e["diff_boot"], e["boot_lo"], e["boot_hi"], {1: "+", -1: "-", 0: "."}[e["robust"]]))
            print("     %-7s" % sn, " | ".join(cells))
sect("2b. spell gain on TYPO slice (MRR dense_s - dense_raw), and clean slice")
for sl in ["typo", "clean"]:
    for sp, sn in SPL:
        cells = []
        for m in modes:
            L = ret[m][sp]["levels"]
            cells.append("%s: %+.3f" % (m, L["dense_s"][sl]["Mrr"] - L["dense_raw"][sl]["Mrr"]))
        print("  %-5s %-7s" % (sl, sn), " | ".join(cells))
sect("2c. tuned hybrid alpha (z-score weight of DENSE; 1.0 = dense only) on A tune seeds")
for m in modes:
    p = ret[m]["picks"]
    print("  %-8s z_alpha raw/spell %s  (mix25 MRR tune grid spell: %s)" % (m, p["z_alpha"], [round(x, 3) for x in ret[m]["tune_table"]["spell"]["z_mrr"]]))
sect("3. g1 templated corpora A,B via the h13 bridge engine (all queries; hit1/MRR/F1@3) per embedder")
for tag in "AB":
    print("-- corpus", tag)
    for sysn in ["dense_raw", "dense_s", "bm25_s", "hybz_raw", "hybz_s", "hybrrf_s", "rerank_s"]:
        row = []
        for m in modes:
            c = ret[m]["g1_bridge"][tag]["levels"][sysn]["all"]
            row.append("%s: %s/%s/%s" % (m, f(c["Hit1"]), f(c["Mrr"]), f(c["F1at3"])))
        print("  %-9s" % sysn, " | ".join(row))
    ca = ret["mini"]["g1_bridge"][tag]["compare_all"]
    print("   (compare_all entry keys:", list(ca[0].keys()) if ca else None, ")")
    for a, b in [("hybz_s", "dense_s"), ("dense_s", "dense_raw"), ("bm25_s", "dense_s")]:
        cells = []
        for m in modes:
            es = [x for x in ret[m]["g1_bridge"][tag]["compare_all"] if x.get("a") == a and x.get("b") == b]
            cells.append("%s: %s" % (m, " ".join("%s %+.3f%s" % (e.get("metric"), e.get("diff", e.get("mean", float("nan"))), "*" if e.get("holm", 1) < .05 else "") for e in es if e.get("metric") in ("Mrr", "F1at3", "Hit1"))))
        print("   contrast %s-%s: " % (a, b), " | ".join(cells))

g = J("h17_g1.json")
sect("4. g1 A-D with the g1_retrieval engine: levels (all held-out queries; seeds as labelled)")
print("alpha picks: hash raw/spell", g["tune_A_tune_concepts"]["hash"]["alpha_raw"], g["tune_A_tune_concepts"]["hash"]["alpha_spell"], "| mini", g["tune_A_tune_concepts"]["mini"]["alpha_raw"], g["tune_A_tune_concepts"]["mini"]["alpha_spell"])
for lab, v in g["eval"].items():
    print("--", lab)
    for sysn in ["bm25_raw", "bm25_s", "hash:dense_raw", "hash:dense_s", "hash:hyb_s", "mini:dense_raw", "mini:dense_s", "mini:hyb_s"]:
        c = v["levels"][sysn]
        print("   %-15s hit1 %.3f MRR %.3f F1@3 %.3f P@3 %.3f R@3 %.3f MRRtypo %.3f MRRclean %.3f" % (sysn, c["hit1"], c["mrr"], c["f13"], c["p3"], c["r3"], c["mrr_typo"], c["mrr_clean"]))
sect("4b. g1 A-D contrasts (diff [95% t-CI] holm-sig): metric mrr / hit1 / f13 / mrr_typo")
keyc = [("dense_s", "dense_raw"), ("hyb_s", "dense_s"), ("hyb_raw", "dense_raw")]
for lab, v in g["eval"].items():
    print("--", lab)
    for m in ["hash", "mini"]:
        for a, b in keyc:
            row = []
            for met in ["mrr", "hit1", "f13", "mrr_typo"]:
                e = [x for x in v["contrasts"] if x["a"] == "%s:%s" % (m, a) and x["b"] == "%s:%s" % (m, b) and x["metric"] == met][0]
                row.append("%s %+.3f[%+.3f,%+.3f]%s" % (met, e["diff"], e["lo"], e["hi"], {"WIN": "+", "LOSS": "-", "ns": "."}[e["sig"]]))
            print("   %-5s %-12s-%-10s" % (m, a, b), " ".join(row))
    for a, b in [("bm25_s", "hash:dense_s"), ("bm25_s", "mini:dense_s"), ("hash:hyb_s", "bm25_s"), ("mini:hyb_s", "bm25_s"), ("mini:dense_s", "hash:dense_s"), ("mini:hyb_s", "hash:hyb_s")]:
        row = []
        for met in ["mrr", "hit1", "f13", "mrr_typo"]:
            e = [x for x in v["contrasts"] if x["a"] == a and x["b"] == b and x["metric"] == met][0]
            row.append("%s %+.3f[%+.3f,%+.3f]%s" % (met, e["diff"], e["lo"], e["hi"], {"WIN": "+", "LOSS": "-", "ns": "."}[e["sig"]]))
        print("   %-13s-%-13s" % (a, b), " ".join(row))
sect("5. context frontier (spell queries; 30% exact-duplicate store; per-concept units): F1/chars(e = TOST +-.01 equivalent to best-F1 policy), dedup effect")
ctx = g["ctx"]
for tag in "ABCD":
    for m in ["mini", "hash"]:
        for dd in [False, True]:
            rows = [r for r in ctx if r.get("corpus") == tag and r.get("model") == m and r.get("dedup") == dd and "policy" in r]
            best = rows[0]["best_policy"]
            sel = ["fixed k=1", "fixed k=2", "fixed k=3", "fixed k=6", "adaptive drop 0.02", "adaptive drop 0.05", "adaptive drop 0.10", "adaptive drop 0.20", "byte budget 400", "byte budget 900"]
            s = "  ".join("%s:%.3f/%d%s" % (r["policy"].replace("fixed k=", "k").replace("adaptive drop ", "d").replace("byte budget ", "B"), r["F1"], r["chars"], "e" if r["equiv_001"] else "") for r in rows if r["policy"] in sel)
            print(" %s %-4s %-8s best=%-20s %s" % (tag, m, "dedup" if dd else "no-dedup", best, s))
    for m in ["mini", "hash"]:
        de = [r for r in ctx if r.get("corpus") == tag and r.get("model") == m and "dedup_effect_policy" in r]
        print("   dedup effect dF1 (dedup - none) %s:" % m, " ".join("%s %+.3f[%+.3f,%+.3f]" % (r["dedup_effect_policy"], r["dF1"], r["ci"][0], r["ci"][1]) for r in de))

sect("6. END-TO-END routing (H2000 means; diff vs same guarded front; * Holm win, ! Holm loss)")
rt = {m: J("h17_route_%s.json" % m) for m in modes if os.path.exists(os.path.join(R, "h17_route_%s.json" % m))}
W = ["T12", "T12-drift", "Tctx", "T48", "T12-noctx", "ctl-irrel", "ctl-oneoff", "Tqa"]
arms = ["safe|dense_s", "safe|bm25_s", "safe|hybz_s", "g1|dense_s", "g1|bm25_s", "g1|hybz_s", "ctx|dense_s", "ctx|hybz_s", "safe|oracle"]
print("worlds:", W)
for sp in ["A_held_31_60", "A_confirm_61_90", "B_confirm_61_90"]:
    print("--", sp)
    for m in rt:
        L = rt[m][sp]["levels"]
        print("  [%s] guarded level  " % m, " ".join("%.4f" % L["guarded"][w]["H2000"] for w in W))
        vs = {(x["arm"], x["world"]): x for x in rt[m][sp]["vs_same_front"] if x["metric"] == "H2000"}
        for a in arms:
            if a not in L:
                continue
            cells = []
            for w in W:
                x = vs.get((a, w))
                if x is None:
                    cells.append("   n/a")
                else:
                    cells.append("%+.3f%s" % (x["diff"], "*" if x["win"] else ("!" if x["loss"] else " ")))
            ip = L[a]["T12"]
            print("    %-13s" % a, " ".join(cells), "| T12 vote %.2f inj_prec %.2f anchor_acc %.2f" % (ip["vote_frac"], ip["inj_prec"], ip["anchor_acc"]))
sect("6b. retrieval variants vs safe|dense_s list (H2000): max |d| and Holm wins/losses across 8 worlds")
for m in rt:
    for sp in ["A_held_31_60", "A_confirm_61_90", "B_confirm_61_90"]:
        xs = [x for x in rt[m][sp]["vs_safe_dense_s"] if x["metric"] == "H2000"]
        out = []
        for a in ["bm25_s", "hybz_s", "hybz_raw", "dense_raw"]:
            for g_ in ["safe", "g1", "ctx"]:
                e = [x for x in xs if x["arm"] == "%s|%s" % (g_, a)]
                if e:
                    out.append("%s|%s: max|d| %.3f w%d l%d" % (g_, a, max(abs(x["diff"]) for x in e), sum(x["win"] for x in e), sum(x["loss"] for x in e)))
        print(" [%s] %s: " % (m, sp), "; ".join(out))
sect("7. CompactRouter (u16/q16) vs f64 recommended router with kn-anchor features from each embedder; H2000 TOST +-0.005, Holm over 8 worlds")
for m in rt:
    ct = rt[m]["compact_tost"]
    for lab, cell in ct.items():
        n_ok = n_all = 0
        worst = 0.0
        det = []
        subj = {k: v for k, v in cell.items() if k != "diag_recF64"}
        for k, v in subj.items():
            if not k.endswith("compact16"):
                continue
            n_all += 1
            n_ok += v["all_equiv95"]
            for w in v["worlds"]:
                worst = max(worst, abs(w["diff"]))
            if not v["all_equiv95"]:
                det.append("%s: %s" % (k, ",".join("%s %+.4f[%+.4f,%+.4f]" % (w["world"], w["diff"], w["lo"], w["hi"]) for w in v["worlds"] if not w["equiv95"])))
        wci = max(max(abs(w["lo"]), abs(w["hi"])) for k, v in subj.items() if k.endswith("compact16") for w in v["worlds"])
        f64ok = all(v["all_equiv95"] for k, v in subj.items() if k.endswith("compactF64"))
        print("  [%s] %-16s compact16 ALL-8-equiv in %d/%d arms; max |diff| %.4f; widest 95%%-CI end %.4f; compactF64 all-equiv %s %s" % (m, lab, n_ok, n_all, worst, wci, f64ok, ("NOT: " + " ; ".join(det)) if det else ""))
    d = ct["A_pooled_31_90"]["diag_recF64"]
    for arm in ["safe|dense_s", "g1|dense_s", "g1|hybz_s"]:
        print("     [%s] %s T12: anchor_acc %.3f found %.3f vote %.3f inj_prec %.3f H2000(recF64) %.4f" % (m, arm, d[arm]["T12"]["anchor_acc"], d[arm]["T12"]["found_frac"], d[arm]["T12"]["vote_frac"], d[arm]["T12"]["inj_prec"], d[arm]["T12"]["H2000"]))
