"""k4-2 prep: build the REAL-structure link graph and the multi-hop QUESTION PAIRS (labels from document structure, never from a generator).
World A: sections / adjacency / explicit markdown links / Cargo.toml dependencies / cross-crate references (py/k4_build.py -> results/k4_struct_A.json).
World B: adjacency inside a file (+ shared rare terms); no section / dependency structure is reconstructed (B reads live files).
Pair types (GT = P1 and the hop-2 target set T2; P1 is a paragraph whose held-out sentence s1 is part of the question, as in the h10 S protocol):
  ADJ     q = s1                           T2 = {next paragraph of the same section / same file}          (neighbour-paragraph completion)
  SECNEXT q = s1 + heading(next section)   T2 = paragraphs of the next section of the same file             (A only)
  DEP     q = s1 + Cargo description(Y)    T2 = all docs of crate Y, X depends on Y in Cargo.toml           (A only)
  XREF    q = s1 + Cargo description(Y)    T2 = all docs of crate Y, P1 links to Y (crates.io/docs.rs/github path)  (A only; re-embedded P1)
  XLINK   q = s1 + heading(target section) T2 = target section, explicit [..](#anchor)/(file.md#anchor) link  (A only; re-embedded P1)
Writes results/k4_mh_pairs_{A,B}.json and results/k4_embed_req.json (texts to embed with py/k4_embed.py in the cognee venv)."""
import collections, hashlib, json, os, re, sys, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
GENERIC = set("""license licence installation install usage features feature contributing contribute example examples table of contents quick start quickstart getting started overview introduction documentation docs changelog changes history readme about requirements prerequisites testing tests test building build running run support faq todo credits acknowledgments acknowledgements authors author contact links resources reference api summary background motivation why configuration config options notes note badges status roadmap security versioning versions development dependencies dependency community sponsors donate the what how when where who this that""".split())
WORD = re.compile(r"[A-Za-z]+")


def hnum(*a):
    return int(hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:10], 16)


def pool_of(i):
    return ["tune", "held", "confirm"][hnum("mhpool", i) % 3]


def sentences(p):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if s.strip()]


def cname(g):
    return re.sub(r"-\d+\.\d+.*$", "", g)


def good_heading(hd):
    if not hd:
        return False
    hw = WORD.findall(hd)
    return 2 <= len(hd.split()) <= 8 and len(hw) >= 2 and not all(w.lower() in GENERIC for w in hw) and any(len(w) >= 4 for w in hw) and hd[0].isalpha()


def main():
    reqs = []  # texts to embed

    def R(t):
        reqs.append(t)
        return len(reqs) - 1

    out = {}
    # ------------------------------------------------------------------ world A
    C = json.load(open(os.path.join(RES, "h10_corpus_A.json"), encoding="utf8"))
    S = json.load(open(os.path.join(RES, "k4_struct_A.json")))
    docs, qs = C["docs"], C["queries"]
    sd = S["docs"]; secs = S["secs"]; deps = S["deps"]
    desc = deps["__desc__"]
    byname = collections.defaultdict(list)
    for i, d in enumerate(docs):
        byname[cname(d["src"])].append(i)
    descname = {}
    for g, t in desc.items():
        if t and len(t.split()) >= 5 and cname(g) not in descname:
            descname[cname(g)] = t
    sdocs = {q["rel"][0]: q for q in qs if q["fam"] == "S" and len(q["rel"]) == 1}
    # first group per crate name that has docs
    pairs = []
    rnd = random.Random(5)
    for d, q in sorted(sdocs.items()):
        s1 = q["text"]; g = docs[d]["src"]; me = cname(g)
        sec, o = sd[d]["sec"], sd[d]["ord"]
        # ADJ: next paragraph of the same section
        if sec >= 0 and o + 1 < len(secs[sec]["docs"]):
            p2 = secs[sec]["docs"][o + 1]
            pairs.append(dict(type="ADJ", p1=d, t2=[p2], q=s1, s1=s1, c2=None, pool=pool_of(d)))
        # SECNEXT: next section in the same file with a good heading
        if sec >= 0 and sec + 1 < len(secs) and secs[sec + 1]["grp"] == secs[sec]["grp"] and secs[sec + 1]["file"] == secs[sec]["file"] and good_heading(secs[sec + 1]["head"]):
            nx = secs[sec + 1]
            pairs.append(dict(type="SECNEXT", p1=d, t2=nx["docs"], q=s1 + " " + nx["head"], s1=s1, c2=nx["head"], pool=pool_of(d)))
        # DEP
        dl = [y for y in deps.get(g, []) if y != me and y in descname and y in byname]
        if dl:
            y = dl[hnum("dep", d) % len(dl)]
            pairs.append(dict(type="DEP", p1=d, t2=byname[y], q=s1 + " " + descname[y], s1=s1, c2=descname[y], pool=pool_of(d), y=y))
    # XREF (cross-crate explicit links) : P1 not necessarily an S doc -> held out sentence, doc re-embedded
    names = set(byname)
    seen = set()
    for i, d in enumerate(sd):
        me = cname(docs[i]["src"])
        for l in d["raw"]:
            for seg in re.split(r"[/#?]", l):
                y = seg.lower().replace("_", "-")
                if y in names and y != me and not me.startswith(y) and not y.startswith(me) and y in descname and (i, y) not in seen:
                    seen.add((i, y))
    for (i, y) in sorted(seen):
        if i in sdocs:
            s1 = sdocs[i]["text"]; newtxt = None
        else:
            ss = sentences(docs[i]["text"])
            c = [k for k, x in enumerate(ss) if 8 <= len(x.split()) <= 30 and x[0].isupper() and y not in x.lower().replace("_", "-")]
            if len(ss) < 3 or not c:
                continue
            k = c[hnum("x", i, y) % len(c)]
            s1 = ss[k]; newtxt = " ".join(ss[:k] + ss[k + 1:])
            if len(newtxt) < 50:
                continue
        pairs.append(dict(type="XREF", p1=i, t2=byname[y], q=s1 + " " + descname[y], s1=s1, c2=descname[y], pool=pool_of(i), y=y, newtxt=newtxt))
    # XLINK (explicit section/file links inside a crate)
    for i, d in enumerate(sd):
        for kind, t in d["tgt"]:
            if kind != "sec" or not good_heading(secs[t]["head"]):
                continue
            if i in sdocs:
                s1 = sdocs[i]["text"]; newtxt = None
            else:
                ss = sentences(docs[i]["text"])
                c = [k for k, x in enumerate(ss) if 8 <= len(x.split()) <= 30 and x[0].isupper()]
                if len(ss) < 3 or not c:
                    continue
                k = c[hnum("xl", i, t) % len(c)]
                s1 = ss[k]; newtxt = " ".join(ss[:k] + ss[k + 1:])
                if len(newtxt) < 50:
                    continue
            pairs.append(dict(type="XLINK", p1=i, t2=secs[t]["docs"], q=s1 + " " + secs[t]["head"], s1=s1, c2=secs[t]["head"], pool=pool_of(i), tsec=t, newtxt=newtxt))
    for p in pairs:
        p["rq"] = R(p["q"]); p["rs1"] = R(p["s1"]); p["rc2"] = R(p["c2"]) if p["c2"] else -1
        p["rdoc"] = R(p["newtxt"]) if p.get("newtxt") else -1
    cnt = collections.Counter((p["type"], p["pool"]) for p in pairs)
    print("A pairs", dict(cnt))
    out["A"] = pairs
    # ------------------------------------------------------------------ world B
    CB = json.load(open(os.path.join(RES, "h10_corpus_B.json"), encoding="utf8"))
    dB, qB = CB["docs"], CB["queries"]
    pairsB = []
    for q in qB:
        if q["fam"] != "S" or len(q["rel"]) != 1:
            continue
        d = q["rel"][0]
        if d + 1 < len(dB) and dB[d + 1]["src"] == dB[d]["src"] and dB[d + 1]["file"] == dB[d]["file"] and dB[d]["file"] != "(doc comment)":
            pairsB.append(dict(type="ADJ", p1=d, t2=[d + 1], q=q["text"], s1=q["text"], c2=None, pool="confirm", rq=R(q["text"]), rs1=-1, rc2=-1, rdoc=-1))
    print("B pairs", len(pairsB))
    out["B"] = pairsB
    json.dump(out, open(os.path.join(RES, "k4_mh_pairs.json"), "w"))
    json.dump(reqs, open(os.path.join(RES, "k4_embed_req.json"), "w"))
    print("embed requests", len(reqs))


if __name__ == "__main__":
    main()
