"""k4-3 prep: STALENESS / CONTRADICTION between versions of the same source (real README paragraphs of every on-disk version of 214 crates, results/k4_versions.json).
Structural key = (crate, section heading slug, paragraph ordinal in the section).  A 'changed fact' query: a sentence of the LATEST version's paragraph that contains a token
(number | identifier-like capitalised word) that does NOT occur in any older version of the same key; the changed tokens are masked out of the query (the user does not know them);
the answer = the masked token(s); a retrieved paragraph answers correctly iff it contains every masked token.  Older versions of the key (same sentence minus the new token) are
near-identical to the query, so flat retrieval cannot tell versions apart lexically -> the question is how often it returns the stale one and what a staleness signal buys.
Controls: STABLE keys (identical text in all versions: any version is right), ONLYOLD keys (paragraph disappeared from the latest version; the correct answer is the old one).
Writes results/k4_vers_q.json and results/k4_vers_req.json (texts to embed)."""
import collections, hashlib, json, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
NUM = re.compile(r"\d[\d._]*\d|\d")
ENT = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")


def hnum(*a):
    return int(hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:10], 16)


def sentences(p):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if s.strip()]


def pool_of(crate):
    h = hnum("vpool", crate) % 5
    return "tune" if h == 0 else ("held" if h in (1, 2) else "confirm")


def main():
    R = json.load(open(os.path.join(RES, "k4_versions.json"), encoding="utf8"))
    for i, r in enumerate(R):
        r["id"] = i
    by = collections.defaultdict(dict)  # key -> vord -> row
    for r in R:
        by[r["key"]][r["vord"]] = r
    latest = {}
    for r in R:
        latest[r["crate"]] = max(latest.get(r["crate"], -1), r["vord"])
    reqs = [r["text"] for r in R]
    Q = []
    for key, vs in sorted(by.items()):
        crate = key.split("|")[0]
        lv = latest[crate]
        if len(vs) < 2:
            continue
        texts = {v: r["text"] for v, r in vs.items()}
        if lv in vs:
            new = vs[lv]
            old = [r for v, r in vs.items() if v != lv]
            if all(o["text"] == new["text"] for o in old):
                kind = "STABLE"
                ss = sentences(new["text"])
                ss = [s for s in ss if 6 <= len(s.split()) <= 35]
                if not ss:
                    continue
                s = ss[hnum("st", key) % len(ss)]
                Q.append(dict(kind=kind, key=key, crate=crate, gold=new["id"], text=s, answer=[], stale=[], pool=pool_of(crate)))
                continue
            oldtok = set()
            for o in old:
                oldtok |= set(NUM.findall(o["text"])) | set(ENT.findall(o["text"]))
            for s in sentences(new["text"]):
                if not (6 <= len(s.split()) <= 40):
                    continue
                toks = [t for t in NUM.findall(s) if t not in oldtok]
                ents = [t for t in ENT.findall(s) if t not in oldtok and not s.startswith(t)]
                if not (toks or ents):
                    continue
                ans = toks if toks else ents
                q = s
                for t in ans:
                    q = q.replace(t, " ")
                q = re.sub(r"\s+", " ", q).strip()
                if len(q.split()) < 5:
                    continue
                stale = [o["id"] for o in old if any(t not in o["text"] for t in ans)]
                if not stale:
                    continue
                Q.append(dict(kind="NUM" if toks else "ENT", key=key, crate=crate, gold=new["id"], text=q, answer=ans, stale=stale, pool=pool_of(crate)))
        else:
            # ONLYOLD: paragraph exists in older versions only
            vmax = max(vs)
            old = vs[vmax]
            ss = [s for s in sentences(old["text"]) if 6 <= len(s.split()) <= 35]
            if not ss:
                continue
            s = ss[hnum("oo", key) % len(ss)]
            Q.append(dict(kind="ONLYOLD", key=key, crate=crate, gold=old["id"], text=s, answer=[], stale=[], pool=pool_of(crate)))
    for q in Q:
        q["rq"] = len(reqs)
        reqs.append(q["text"])
    c = collections.Counter((q["kind"], q["pool"]) for q in Q)
    print("queries", dict(c), "docs", len(R))
    json.dump(dict(rows=R, queries=Q), open(os.path.join(RES, "k4_vers_q.json"), "w"))
    json.dump(reqs, open(os.path.join(RES, "k4_vers_req.json"), "w"))


if __name__ == "__main__":
    main()
