"""h20 SYNTHETIC domain corpus generator (clearly labelled SYNTHETIC; used ONLY for domains with no real text on disk:
scientist, engineer, business analyst, executive, financial analyst, commodities analyst, economist, linguist, psychologist, historian).
Seeded and deterministic.  Documents are word-salad paragraphs of real English domain words (so MiniLM gives them meaningful vectors) with:
  * 8 sub-topics per domain (10 words each) + domain-general words + CROSS-DOMAIN CONFUSABLE words shared by >= 3 domains,
  * with prob p_cross a document's secondary sub-topic comes from ANOTHER (sibling) domain (borderline docs),
  * a document-unique pseudo-word entity in two sentences (hapax; BM25 can pin it): the held-out query sentence contains it w.p. H_Q (knob).
Held-out sentence = query, paragraph minus that sentence = the gold doc (same protocol as h10 S queries).
Usage: python py/h20_syn.py  -> results/h20_corpus_syn10.json   {docs:[{id,domain,src,text,query}], meta}
"""
import hashlib, json, os, random

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
N_PER = 320
H_Q = 0.35  # probability that the query sentence carries the document's unique entity token
P_CROSS = 0.15

V = {
 "scientist": ["hypothesis experiment control replication variance sample;spectroscopy wavelength photon emission absorption laser detector;genome sequencing protein enzyme mutation allele expression;orbit galaxy telescope redshift nebula exoplanet luminosity;catalyst molecule reaction synthesis solvent yield isotope;ecosystem species habitat biodiversity predator migration climate;quantum particle entanglement boson symmetry lattice;peer review preprint dataset reproducibility calibration uncertainty"],
 "engineer": ["tolerance torque load beam stress fatigue weld;circuit voltage resistor capacitor transistor amplifier firmware;throughput latency bandwidth queue scheduler cache pipeline;hydraulic valve pump turbine pressure flow gasket;compiler runtime memory pointer thread deadlock kernel;specification requirements prototype integration commissioning acceptance;thermal heatsink conduction coolant radiator insulation;reliability redundancy failover maintenance inspection safety margin"],
 "business analyst": ["stakeholder requirements backlog workflow process mapping gap;dashboard kpi funnel conversion churn cohort retention;swimlane bpmn user story acceptance criteria traceability;vendor procurement contract sla onboarding;forecast pipeline quota territory segmentation persona;benchmark competitor positioning pricing tier bundle;change request scope baseline milestone governance;reporting data warehouse etl metric definition lineage"],
 "executive": ["strategy vision mission board governance shareholder;acquisition merger synergy integration restructuring divestiture;leadership succession culture alignment talent accountability;quarterly guidance investor earnings roadmap priorities;enterprise transformation portfolio sponsorship steering;brand reputation crisis communication stakeholder trust;operating model budget headcount capital allocation;regulatory compliance audit committee disclosure risk appetite"],
 "financial analyst": ["valuation discounted cashflow terminal multiple ebitda comparables;portfolio diversification sharpe beta alpha drawdown rebalancing;bond duration convexity coupon yield spread covenant;equity dividend buyback earnings guidance accretion dilution;balance sheet leverage liquidity working capital solvency;derivative option hedge swap futures volatility delta;budget variance forecast scenario sensitivity planning;credit rating default recovery underwriting securitization"],
 "commodities analyst": ["crude brent barrel refinery inventory contango backwardation;copper aluminum smelter ore concentrate treatment charge;wheat corn soybean harvest acreage yield drought;natural gas lng pipeline storage injection heating demand;gold silver bullion palladium safe haven central bank reserves;freight tanker bulk carrier charter rate port congestion;opec quota production cut spare capacity sanction embargo;futures curve spread arbitrage physical delivery warehouse warrant"],
 "economist": ["inflation deflation monetary policy interest rate central bank liquidity;gdp output gap recession expansion business cycle productivity;unemployment labor participation wage rigidity phillips;trade tariff exchange rate current account comparative advantage;fiscal deficit stimulus multiplier austerity sovereign debt;elasticity equilibrium marginal utility consumer surplus welfare;regression instrument causal identification panel endogeneity;inequality gini mobility human capital education returns"],
 "linguist": ["phoneme morpheme syntax semantics pragmatics lexicon grammar;dialect sociolinguistics register code switching prestige variation;etymology cognate proto language sound change borrowing;corpus collocation concordance frequency tokenization annotation;ergative agreement case marking argument structure valency;prosody intonation stress tone vowel consonant articulation;bilingual acquisition critical period immersion fluency transfer;discourse anaphora ellipsis presupposition implicature coherence"],
 "psychologist": ["cognition attention memory perception working recall encoding;anxiety depression therapy cognitive behavioral exposure remission;attachment development adolescence parenting temperament bonding;personality extraversion neuroticism trait inventory factor;conformity obedience bystander persuasion heuristic attitude;reinforcement conditioning extinction reward punishment habit;bias priming anchoring framing heuristic judgment decision;sleep stress cortisol resilience trauma coping wellbeing"],
 "historian": ["archive manuscript chronicle source provenance historiography;empire dynasty conquest colonization treaty diplomacy succession;revolution reform uprising abolition suffrage movement;medieval feudal monastery guild crusade plague peasantry;industrialization urbanization railway factory proletariat migration;ancient republic senate legion aqueduct inscription papyrus;cold war nationalism decolonization partition propaganda;memory commemoration oral history testimony archive interpretation"],
}
SHARED = "risk model rate value growth structure system return cycle signal pattern bias market index transfer stress balance margin capital key frame base field form order power term case process measure curve scale shift limit source trend level gap".split()
GENERAL = "analysis evidence framework approach result significant increase decrease relationship factor effect impact context measure assessment review study report practice standard method outcome policy condition difference period regional recent historical systematic empirical reported observed estimated considered".split()
FILL = "the of and in to a is that for with as on by this are it an from which was be has at or can may not more also between during under about over after across within through such these their its our when where however thus often typically largely rarely mostly clearly broadly".split()
DOMS = list(V.keys())
SUB = {d: [x.split() for x in V[d][0].split(";")] for d in DOMS}
SYL = ["ka", "ro", "vel", "mi", "tan", "sor", "bel", "dru", "fen", "gal", "hup", "jor", "lim", "nax", "pel", "qui", "rus", "tev", "wal", "zon"]


def ent(r):
    return "".join(r.choice(SYL) for _ in range(3)).capitalize()


def make_doc(dom, idx, seed=20260102):
    r = random.Random(int(hashlib.sha1(("%s|%d|%d" % (dom, idx, seed)).encode()).hexdigest()[:12], 16))
    main = r.choice(SUB[dom])
    if r.random() < P_CROSS:
        od = r.choice([d for d in DOMS if d != dom])
        second = r.choice(SUB[od])
        cross = od
    else:
        second = r.choice(SUB[dom])
        cross = None
    dgen = [w for s in SUB[dom] for w in s]
    e = ent(r)
    ns = r.choice([3, 4, 4, 5])
    sents = []
    for i in range(ns):
        n = r.randint(11, 17)
        ws = []
        for _ in range(n):
            u = r.random()
            if u < 0.30:
                ws.append(r.choice(main))
            elif u < 0.38:
                ws.append(r.choice(second))
            elif u < 0.50:
                ws.append(r.choice(dgen))
            elif u < 0.62:
                ws.append(r.choice(SHARED))
            elif u < 0.74:
                ws.append(r.choice(GENERAL))
            else:
                ws.append(r.choice(FILL))
        ent_pos = i in (0, ns - 1 if ns > 2 else 1)
        if ent_pos:
            ws.insert(r.randrange(2, len(ws)), e)
        s = " ".join(ws)
        sents.append((s[0].upper() + s[1:] + ".", ent_pos))
    withe = [i for i, (_, p) in enumerate(sents) if p]
    wo = [i for i, (_, p) in enumerate(sents) if not p]
    cut = r.choice(withe) if (r.random() < H_Q or not wo) else r.choice(wo)
    query = sents[cut][0]
    doc = " ".join(s for i, (s, _) in enumerate(sents) if i != cut)
    return {"domain": dom, "src": "syn:%s:%d" % (dom, idx), "text": doc, "query": query, "cross": cross, "ent_in_query": cut in withe}


def main():
    docs = []
    for dom in DOMS:
        for i in range(N_PER):
            docs.append(make_doc(dom, i))
    for i, d in enumerate(docs):
        d["id"] = i
    meta = {"tag": "syn10", "n": len(docs), "H_Q": H_Q, "P_CROSS": P_CROSS, "per_domain": N_PER, "SYNTHETIC": True,
            "sha1": hashlib.sha1("\n".join(d["text"] for d in docs).encode()).hexdigest()}
    json.dump({"docs": docs, "meta": meta}, open(os.path.join(RES, "h20_corpus_syn10.json"), "w", encoding="utf8"))
    print(json.dumps(meta))
    print(docs[0]["text"], "\nQ:", docs[0]["query"])


if __name__ == "__main__":
    main()
