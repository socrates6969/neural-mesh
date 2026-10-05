"""k4 fixture builder (REAL structure from on-disk docs; read-only, nothing executed, no network).
Outputs: results/k4_struct_A.json  (world A paragraphs: section id/heading/ordinal, explicit md links resolved to (file|section), crate deps from Cargo.toml)
         results/k4_versions.json  (multi-version crates: every version's README paragraphs with structural key)
World A is rebuilt with the SAME code/seeds as py/h10_corpus.py and verified against the saved results/h10_corpus_A.json (ids, src, file, text up to the
held-out-sentence edit of S queries)."""
import collections, glob, importlib.util, json, os, re, sys, hashlib
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("h10c", os.path.join(ROOT, "py", "h10_corpus.py"))
h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)


class LStr(str):
    links = ()


LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)[^)]*\)")


def parse_md_links(text):
    """same as h10 parse_md but paragraphs are LStr carrying the raw link targets of the markdown they came from"""
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    lines = text.splitlines()
    secs, head, paras, cur, fence = [], None, [], [], False

    def flush():
        nonlocal cur
        if cur:
            raw = " ".join(cur)
            p = h.clean_inline(raw)
            p = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", p)
            if h.ok_paragraph(p):
                q = LStr(p)
                q.links = tuple(LINK.findall(raw))
                paras.append(q)
            cur = []

    for ln in lines:
        s = ln.strip()
        if s.startswith("```") or s.startswith("~~~"):
            flush(); fence = not fence; continue
        if fence:
            continue
        m = re.match(r"^(#{1,4})\s+(.*?)\s*#*$", s)
        if m:
            flush()
            if head is not None or paras:
                secs.append((head, paras))
            head, paras = h.clean_inline(m.group(2)), []
            continue
        if not s or s.startswith("|") or s.startswith("<") or re.match(r"^[-=*_]{3,}$", s) or s.startswith("[!["):
            flush(); continue
        if re.match(r"^(?:[-*+]|\d+[.)])\s+", s):
            flush()
        if ln.startswith("    ") or ln.startswith("\t"):
            flush(); continue
        cur.append(s)
    flush()
    secs.append((head, paras))
    return secs


def slug(hd):
    s = re.sub(r"[^a-z0-9 _-]", "", (hd or "").lower())
    return s.strip().replace(" ", "-")


def crate_name(g):
    return re.sub(r"-\d+\.\d+.*$", "", g)


def parse_deps(cargo_text):
    deps = set(); sect = None
    for ln in cargo_text.splitlines():
        m = re.match(r"^\s*\[([^\]]+)\]\s*$", ln)
        if m:
            sect = m.group(1).strip()
            m2 = re.match(r"^(?:target\.[^\]]*\.)?(?:dev-|build-)?dependencies\.([A-Za-z0-9_-]+)$", sect)
            if m2:
                deps.add(m2.group(1).replace("_", "-"))
            continue
        if sect and re.match(r"^(?:target\..*\.)?(?:dependencies)$", sect):
            m3 = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=", ln)
            if m3:
                deps.add(m3.group(1).replace("_", "-"))
    return sorted(deps)


def build_A():
    R = glob.glob(os.path.expanduser("~/.cargo/registry/src/*/"))[0]
    h.parse_md = parse_md_links  # build_world calls the module-level name
    files = h.collect_registry()
    docs, secs = h.build_world("A", files, [], max_units=9000, per_file=40, per_group=60, seed=11)
    saved = json.load(open(os.path.join(RES, "h10_corpus_A.json"), encoding="utf8"))
    sd = saved["docs"]
    assert len(sd) == len(docs), (len(sd), len(docs))
    bad = 0; s_edit = 0
    for a, b in zip(sd, docs):
        assert a["src"] == b["src"] and a["file"] == b["file"]
        if a["text"] != b["text"]:
            s_edit += 1
            if len(a["text"]) >= len(b["text"]):
                bad += 1
    print("A rebuilt: docs", len(docs), "S-edited", s_edit, "bad", bad)
    assert bad == 0
    doc_sec = {}
    for si, s in enumerate(secs):
        for o, i in enumerate(s["docs"]):
            doc_sec[i] = (si, o)
    file_secs = collections.defaultdict(dict)
    for si, s in enumerate(secs):
        fl = docs[s["docs"][0]]["file"]
        file_secs[(s["grp"], fl)][slug(s["head"])] = si
    files_in_grp = collections.defaultdict(lambda: collections.defaultdict(list))
    for si, s in enumerate(secs):
        files_in_grp[s["grp"]][docs[s["docs"][0]]["file"].lower()].append(si)
    out_docs = []; nlinks = collections.Counter()
    for d in docs:
        i = d["id"]; si, o = doc_sec.get(i, (-1, 0))
        links = getattr(d["text"], "links", ())
        tgt = []
        for l in links:
            l = l.strip()
            if l.startswith("#"):
                t = file_secs[(d["src"], d["file"])].get(l[1:].lower())
                if t is not None and t != si:
                    tgt.append(["sec", t]); nlinks["anchor"] += 1
            elif not re.match(r"^[a-z]+://|^mailto:", l):
                base = os.path.basename(l.split("#")[0]).lower()
                if base.endswith(".md") and base != d["file"].lower() and base in files_in_grp[d["src"]]:
                    a = l.split("#")[1].lower() if "#" in l else None
                    ts = files_in_grp[d["src"]][base]
                    t = None
                    if a is not None:
                        for tt in ts:
                            if slug(secs[tt]["head"]) == a:
                                t = tt
                    tgt.append(["sec", t] if t is not None else ["file", base]); nlinks["file"] += 1
        out_docs.append(dict(id=i, sec=si, ord=o, tgt=tgt, nlinks_raw=len(links), raw=[l for l in links if re.match(r'^https?://', l)][:6]))
    nsee = sum(1 for d in docs if re.search(r"\b[Ss]ee (the |our |also )?[A-Z`\[]", d["text"]))
    print("links:", dict(nlinks), "docs with raw links", sum(1 for d in out_docs if d["nlinks_raw"]), "see-refs", nsee)
    names = collections.defaultdict(list)
    for g in sorted(set(d["src"] for d in docs)):
        names[crate_name(g)].append(g)
    deps = {"__desc__": {}, "__cat__": {}}
    for g in names.values():
        for gg in g:
            p = os.path.join(R, gg, "Cargo.toml")
            try:
                t = open(p, encoding="utf8", errors="ignore").read()
            except Exception:
                continue
            deps[gg] = [x for x in parse_deps(t) if x in names]
            m = re.search(r'^description\s*=\s*"([^"]*)"', t, flags=re.M)
            deps["__desc__"][gg] = m.group(1) if m else ""
            cm = re.search(r"^categories\s*=\s*\[([^\]]*)\]", t, flags=re.M)
            deps["__cat__"][gg] = re.findall(r'"([^"]+)"', cm.group(1)) if cm else []
    print("crates with in-store deps:", sum(1 for k, v in deps.items() if not k.startswith("__") and v), "of", len(names))
    secs_out = [dict(head=s["head"], grp=s["grp"], docs=s["docs"], file=docs[s["docs"][0]]["file"]) for s in secs]
    json.dump(dict(docs=out_docs, secs=secs_out, deps=deps), open(os.path.join(RES, "k4_struct_A.json"), "w"))


def vkey(v):
    return tuple(int(x) if x.isdigit() else 0 for x in re.split(r"[.+-]", v)[:4])


def build_versions():
    R = glob.glob(os.path.expanduser("~/.cargo/registry/src/*/"))[0]
    by = collections.defaultdict(list)
    for c in sorted(os.listdir(R)):
        m = re.match(r"^(.*?)-(\d+\.\d+\.\d+.*)$", c)
        if m and os.path.isdir(os.path.join(R, c)):
            by[m.group(1)].append((m.group(2), c))
    rows = []; ncr = 0
    for name, vs in sorted(by.items()):
        if len(vs) < 2:
            continue
        vs.sort(key=lambda x: vkey(x[0]))
        got = {}
        for ver, c in vs:
            for f in os.listdir(os.path.join(R, c)):
                if f.lower() == "readme.md":
                    t = h.read(os.path.join(R, c, f))
                    if not t:
                        continue
                    seen = []
                    for head, paras in h.parse_md(t):
                        for o, p in enumerate(paras[:12]):
                            seen.append((slug(head) or "_", o, str(p)))
                    got[ver] = seen[:80]
        if len(got) < 2:
            continue
        ncr += 1
        order = [v for v, _ in vs if v in got]
        for ver, seen in got.items():
            for hs, o, p in seen:
                rows.append(dict(crate=name, ver=ver, key="%s|%s|%d" % (name, hs, o), head=hs, text=p, vord=order.index(ver), nver=len(order)))
    print("version corpus: crates", ncr, "paragraph rows", len(rows))
    json.dump(rows, open(os.path.join(RES, "k4_versions.json"), "w"))


if __name__ == "__main__":
    build_A(); build_versions()
