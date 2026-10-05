"""h17: markdown delta tables (hash - MiniLM) from results/h17_*.json  (no new computation).  Usage: python py/h17_tables.py > results/h17_tables.md"""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
J = lambda n: json.load(open(os.path.join(R, n), encoding="utf8"))
modes = ["mini", "hash", "hashtok"]
ret = {m: J("h17_retrieval_%s.json" % m) for m in modes}
g = J("h17_g1.json")
rt = {m: J("h17_route_%s.json" % m) for m in modes}
SPL = [("held_A_31_60", "real A held"), ("confirm_A_61_90", "real A confirm"), ("confirm_B_61_90", "real B confirm")]
print("### T1. REAL doc text, doc level (mix25: 75% clean + 25% typo queries). MiniLM | hash(HashEmbedding as shipped) | hashtok(lowercase words) | delta hash-MiniLM | delta hashtok-MiniLM\n")
print("| split | system | metric | MiniLM | hash | hashtok | d(hash) | d(hashtok) |\n|---|---|---|---|---|---|---|---|")
for sp, sn in SPL:
    for sysn in ["dense_raw", "dense_s", "bm25_s", "hybz_s"]:
        for met, mn in [("Hit1", "hit@1"), ("Mrr", "MRR"), ("P3", "P@3"), ("R3", "R@3"), ("F1at3", "F1@3")]:
            v = [ret[m][sp]["levels"][sysn]["mix25"][met] for m in modes]
            print("| %s | %s | %s | %.3f | %.3f | %.3f | %+.3f | %+.3f |" % (sn, sysn, mn, v[0], v[1], v[2], v[1] - v[0], v[2] - v[0]))
print("\n### T2. g1 templated corpora A-D (g1_retrieval engine; hash = lowercase-word feature hashing as in g1; all held-out queries; seeds 31-60 (A) / 61-90)\n")
print("| corpus split | system | hit@1 | MRR | F1@3 | P@3 | R@3 | MRR typo |\n|---|---|---|---|---|---|---|---|")
for lab, v in g["eval"].items():
    for sysn in ["bm25_s", "hash:dense_raw", "hash:dense_s", "hash:hyb_s", "mini:dense_raw", "mini:dense_s", "mini:hyb_s"]:
        c = v["levels"][sysn]
        print("| %s | %s | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (lab, sysn, c["hit1"], c["mrr"], c["f13"], c["p3"], c["r3"], c["mrr_typo"]))
print("\n### T3. END-TO-END routing H2000 (g1 gate, spell queries): arm minus the SAME guarded front; * = Holm win, ! = Holm loss. Cell = MiniLM / hash / hashtok\n")
W = ["T12", "T12-drift", "T48", "T12-noctx", "Tctx", "ctl-irrel", "ctl-oneoff", "Tqa"]
for sp, sn in [("A_held_31_60", "A held 31-60"), ("A_confirm_61_90", "A confirm 61-90"), ("B_confirm_61_90", "B confirm 61-90")]:
    print("\n**%s**\n\n| arm | %s |\n|---|%s" % (sn, " | ".join(W), "---|" * len(W)))
    for a in ["safe|dense_s", "safe|hybz_s", "g1|dense_s", "g1|bm25_s", "g1|hybz_s", "ctx|dense_s", "ctx|hybz_s"]:
        cells = []
        for w in W:
            parts = []
            for m in modes:
                x = [y for y in rt[m][sp]["vs_same_front"] if y["arm"] == a and y["world"] == w and y["metric"] == "H2000"][0]
                parts.append("%+.3f%s" % (x["diff"], "*" if x["win"] else ("!" if x["loss"] else "")))
            cells.append(" / ".join(parts))
        print("| %s | %s |" % (a, " | ".join(cells)))
print("\n### T4. end-to-end: same-gate hybz_s minus dense_s (H2000) and anchor diagnostics (T12, A held)\n")
print("| encoder | gate | A held (T12,drift,T48,noctx) | A conf | B conf | vote share | injected-paragraph relevance | anchor-cluster accuracy |\n|---|---|---|---|---|---|---|---|")
for m in modes:
    for gt in ["safe", "g1"]:
        row = []
        for sp in ["A_held_31_60", "A_confirm_61_90", "B_confirm_61_90"]:
            xs = {x["world"]: x for x in rt[m][sp]["vs_safe_dense_s"] if x["arm"] == "%s|hybz_s" % gt and x["metric"] == "H2000"}
            row.append(" ".join("%+.3f%s" % (xs[w]["diff"], "*" if xs[w]["win"] else "") for w in ["T12", "T12-drift", "T48", "T12-noctx"]))
        d = rt[m]["A_held_31_60"]["levels"]
        print("| %s | %s | %s | %s | %s | dense_s %.2f / hybz_s %.2f | %.2f / %.2f | %.2f / %.2f |" % (m, gt, row[0], row[1], row[2], d["%s|dense_s" % gt]["T12"]["vote_frac"], d["%s|hybz_s" % gt]["T12"]["vote_frac"], d["%s|dense_s" % gt]["T12"]["inj_prec"], d["%s|hybz_s" % gt]["T12"]["inj_prec"], d["%s|dense_s" % gt]["T12"]["anchor_acc"], d["%s|hybz_s" % gt]["T12"]["anchor_acc"]))
print("\n### T5. CompactRouter vs f64 recommended router, kn-anchor (+ kd:<doc> open-vocabulary) features from each encoder's retrieval. H2000 diff compact16 - recF64 [95% CI]; margin +-0.005; E = Holm-TOST and 95%-CI inside; worlds listed T12,drift,Tctx,T48,noctx,ctl-irrel,ctl-oneoff,Tqa\n")
print("| split | encoder | anchor arm | all 8 worlds equivalent | worst |diff| | widest CI end | compactF64 diff exactly 0 |\n|---|---|---|---|---|---|---|")
for lab in ["A_pooled_31_90", "B_confirm_61_90"]:
    for m in modes:
        cell = rt[m]["compact_tost"][lab]
        for k, v in cell.items():
            if k == "diag_recF64" or not k.endswith("|compact16"):
                continue
            base = k[: -len("|compact16")]
            f64 = cell[base + "|compactF64"]
            wd = max(abs(w["diff"]) for w in v["worlds"])
            wc = max(max(abs(w["lo"]), abs(w["hi"])) for w in v["worlds"])
            z = all(abs(w["diff"]) < 1e-12 and abs(w["lo"]) < 1e-12 and abs(w["hi"]) < 1e-12 for w in f64["worlds"])
            print("| %s | %s | %s | %s | %.4f | %.4f | %s |" % (lab, m, base, "yes" if v["all_equiv95"] else "NO", wd, wc, "yes" if z else "no"))
