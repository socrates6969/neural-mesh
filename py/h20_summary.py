"""h20 summary tables from results/h20_persona_{held,confirm}.json and results/h20_retrieval_{held,confirm}.json (+ sens, tune).  python py/h20_summary.py > results/h20_summary.txt"""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
L = lambda f: json.load(open(os.path.join(RES, f)))


def persona():
    print("### A. PERSONA/ROLE routing: H%d mean quality diff vs BEST FAIR generic baseline, paired 95%% CI (held 31-60 | confirm 61-90); * = Holm-significant over 10 worlds x 7 policies" % 1000)
    H, C = L("h20_persona_held.json"), L("h20_persona_confirm.json")
    pols = ["oracle_role", "anchor_role", "learned_role", "joint", "gate_static", "gate_ab", "gate_ab_agent"]
    print("%-10s %-7s " % ("world", "bestfair") + " ".join("%-26s" % p for p in pols))
    for wn in H["worlds"]:
        row = "%-10s %-7s " % (wn, H["worlds"][wn]["best_fair_baseline"])
        for p in pols:
            cs = []
            for S in (H, C):
                c = S["worlds"][wn]["contrasts"]["%s_vs_bestfair_h" % p]
                cs.append("%+.3f%s" % (c["diff"], "*" if c.get("holm_sig") else ("=" if c.get("equiv") else " ")))
            row += "%-26s" % (" | ".join(cs))
        print(row)
    print("(= means TOST-equivalent within +-0.01 and not Holm-significant)")
    print("\nbaselines H (held): ", {wn: {b: round(W["pol"][b]["h"], 3) for b in ("eps", "pts", "kweps", "kwts")} for wn, W in H["worlds"].items() if wn in ("G1-mod", "G2-mod", "ctl-mod")})
    print("ceilings (held): ", {wn: (round(W["ceil_generic_agent"], 3), round(W["ceil_agent_plus_oracle_role"], 3)) for wn, W in H["worlds"].items()})
    print("role-prompted call fraction (confirm): ", {p: {wn: round(W["pol"][p]["role_frac"], 2) for wn, W in C["worlds"].items() if wn in ("G1-mod", "G1-zero", "G1-harm", "G1-strong", "G2-harm")} for p in ("gate_ab", "gate_ab_agent")})
    print("early (first 250 tasks) gate_ab minus bestfair, confirm: ", {wn: round(W["contrasts"]["gate_ab_vs_bestfair_early"]["diff"], 4) for wn, W in C["worlds"].items()})
    t = L("h20_persona_tune.json")
    print("tuned:", {k: v for k, v in t.items() if k != "table"})
    print("holm count confirm:", sum(h["holm_sig"] for h in C["holm"]), "of", len(C["holm"]), " held:", sum(h["holm_sig"] for h in H["holm"]), "of", len(H["holm"]))


def retrieval():
    print("\n### B. DOMAIN-CORPUS RETRIEVAL (hybrid z(.7)+BM25+spell pooled = best fair pooled baseline; frozen domain arm = tune pick)")
    for ph in ("held", "confirm"):
        S = L("h20_retrieval_%s.json" % ph)
        fz = S["tuned"]["frozen"]
        print("-- %s  frozen=%s  families=%s" % (ph, fz, S["tuned"]["families"]))
        print("%-13s %-7s %-7s %-7s %-7s | %-7s %-7s %-7s | %-24s %-24s %-24s %-10s" % ("world", "domOK", "none", "dense+sp", "hyb+sp", "frozen", "oracleM", "nocheck", "dq (frozen-hyb) [CI]", "dhit5", "dmrr", "TOST.01"))
        for wn, W in S["worlds"].items():
            a = W["arms"]
            cq = W["contrasts"]["%s_vs_hyb_spell_q" % fz]
            ch = W["contrasts"]["%s_vs_hyb_spell_hit5" % fz]
            cm = W["contrasts"]["%s_vs_hyb_spell_mrr" % fz]
            print("%-13s %-7s %-7.3f %-7.3f %-7.3f | %-7.3f %-7.3f %-7.3f | %+.4f [%+.4f,%+.4f]   %+.4f [%+.4f,%+.4f]   %+.4f [%+.4f,%+.4f]   %s" % (
                wn, W["domain_mode_enabled"], a["none"]["q"], a["dense_spell"]["q"], a["hyb_spell"]["q"], a[fz]["q"], a["oracle_M"]["q"], a[fz + "#nocheck"]["q"],
                cq["diff"], cq["lo"], cq["hi"], ch["diff"], ch["lo"], ch["hi"], cm["diff"], cm["lo"], cm["hi"], cq["equiv"]))
        print("   retrieval metrics hit5/mrr/off5 (hyb+spell -> frozen -> oracleM):")
        for wn, W in S["worlds"].items():
            a = W["arms"]
            print("   %-13s hit5 %.3f -> %.3f -> %.3f | mrr %.3f -> %.3f -> %.3f | off5 %.3f -> %.3f -> %.3f | dense_raw hit5 %.3f dense+sp %.3f hyb_raw %.3f" % (
                wn, a["hyb_spell"]["hit5"], a[fz]["hit5"], a["oracle_M"]["hit5"], a["hyb_spell"]["mrr"], a[fz]["mrr"], a["oracle_M"]["mrr"], a["hyb_spell"]["off"], a[fz]["off"], a["oracle_M"]["off"], a["dense_raw"]["hit5"], a["dense_spell"]["hit5"], a["hyb_raw"]["hit5"]))
        print("   subsets (frozen minus hyb+spell) q / hit5 for typo | clean | confusable-tertile:")
        for wn, W in S["worlds"].items():
            if wn.endswith("+shuf"):
                continue
            a = W["arms"]
            f = lambda s, k: a[fz]["_sub_" + s][k] - a["hyb_spell"]["_sub_" + s][k]
            print("   %-9s q %+.4f %+.4f %+.4f | hit5 %+.4f %+.4f %+.4f" % (wn, f("typo", "q"), f("clean", "q"), f("conf", "q"), f("typo", "hit5"), f("clean", "hit5"), f("conf", "hit5")))
        print("   Holm (frozen vs none/hyb_spell on q): ", [(h["world"], h["ref"], h["holm_sig"]) for h in S["holm_q_contrasts"] if h["arm"] == fz])
    print("\nsensitivity (held), frozen minus hyb+spell on q:")
    s = L("h20_retrieval_sens_held.json")
    for k, v in s.items():
        print("  %-5s %s" % (k, {w: round(x["frozen_minus_pooled_q"], 4) for w, x in v.items() if not w.endswith("+shuf")}))
    t = L("h20_retrieval_tune.json")
    print("tune top:", t["grid"][:6], "best by family", t["best_by_family"], "pooled tune", t["pooled_tune_q"])


if __name__ == "__main__":
    persona()
    retrieval()
