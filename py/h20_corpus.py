"""h20 corpus builder: DOMAIN-LABELLED REAL TEXT from files already on this disk (no network, no downloads, nothing executed: files are
only read and parsed with `ast` / regex).

Generator R1 ("pydoc9"): docstring prose of installed python packages, domain = package family (provenance label, not a guess):
  numerics_stats  scipy, mpmath, sympy, stdlib statistics
  ml_data         sklearn, pandas
  crypto_security cryptography, jwt, bcrypt, argon2, OpenSSL, pwdlib, asn1crypto
  language_text   inflect, langdetect, py_rust_stemmers, regex, tiktoken, pyparsing, markdown_it, stdlib difflib
  time_decimal    dateutil, pendulum, pytz, humanize, stdlib _pydecimal/fractions/calendar/_pydatetime
  graph_knowledge networkx, rdflib
  web_network     aiohttp, httpx, requests, urllib3, h2, starlette, fastapi, websockets, dns
  db_cloud        sqlalchemy, alembic, redis, boto3(botocore docs strings skipped), tomlkit
  project_research this repo's results/*.md + DESIGN.md prose
Generator R2 ("crates8"): README.md + lib.rs //! docs of the 1200 vendored crates in ~/.cargo/registry/src/*, domain = the crate's own
  Cargo.toml `categories` (first category that maps): cryptography, mathematics+science, text-processing+parser-implementations,
  network-programming, database, date-and-time, compression+encoding, os+hardware-support.
Units = prose paragraphs, 80-600 chars, >= 3 sentences (so a held-out sentence can be the query), filtered like h10 (no code/tables/URLs).
Output: results/h20_corpus_{pydoc9,crates8}.json  {docs:[{id,domain,src,text}], meta}.   Usage: python py/h20_corpus.py
"""
import ast, collections, glob, hashlib, importlib.util, json, os, random, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("h10c", os.path.join(ROOT, "py", "h10_corpus.py"))
h10c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h10c)

HOME = os.path.expanduser("~")
SP1 = HOME + "/.cognee-plugin/venv/Lib/site-packages"
SP2 = HOME + "/AppData/Local/Programs/Python/Python314/Lib/site-packages"
STD = HOME + "/AppData/Local/Programs/Python/Python314/Lib"
CAP_DOMAIN = 700
CAP_FILE = 14

PY9 = {
    "numerics_stats": [(SP2, "scipy"), (SP1, "mpmath"), (SP1, "sympy"), (STD, "statistics.py"), (STD, "math.py"), (STD, "cmath.py"), (STD, "numbers.py")],
    "ml_data": [(SP2, "sklearn"), (SP1, "pandas")],
    "crypto_security": [(SP1, "cryptography"), (SP1, "jwt"), (SP1, "bcrypt"), (SP1, "argon2"), (SP2, "OpenSSL"), (SP1, "pwdlib"), (SP2, "asn1crypto"), (STD, "hashlib.py"), (STD, "hmac.py"), (STD, "secrets.py"), (STD, "ssl.py"), (STD, "random.py"), (STD, "pkcs11"), (SP2, "pkcs11")],
    "language_text": [(SP1, "inflect"), (SP1, "langdetect"), (SP1, "py_rust_stemmers"), (SP1, "regex"), (SP1, "tiktoken"), (SP1, "pyparsing"), (SP1, "markdown_it"), (STD, "difflib.py"), (STD, "re"), (STD, "textwrap.py"), (STD, "string"), (STD, "locale.py"), (STD, "gettext.py"), (STD, "email"), (STD, "html"), (STD, "unicodedata.py"), (STD, "codecs.py"), (STD, "encodings")],
    "time_decimal": [(SP1, "dateutil"), (SP1, "pendulum"), (SP1, "pytz"), (SP1, "humanize"), (STD, "_pydecimal.py"), (STD, "fractions.py"), (STD, "calendar.py"), (STD, "_pydatetime.py"), (STD, "zoneinfo"), (STD, "numbers.py"), (STD, "sched.py"), (STD, "datetime.py"), (STD, "time.py"), (SP2, "dateutil"), (SP2, "pytz")],
    "graph_knowledge": [(SP1, "networkx"), (SP1, "rdflib")],
    "web_network": [(SP1, "aiohttp"), (SP1, "httpx"), (SP1, "requests"), (SP1, "urllib3"), (SP2, "h2"), (SP1, "starlette"), (SP1, "fastapi"), (SP1, "websockets"), (SP1, "dns")],
    "db_cloud": [(SP1, "sqlalchemy"), (SP1, "alembic"), (SP1, "redis"), (SP1, "tomlkit")],
}

CRATE_CAT = [
    ("cryptography", "crypto"),
    ("mathematics", "math_sci"), ("science", "math_sci"), ("simulation", "math_sci"),
    ("text-processing", "text_i18n"), ("internationalization", "text_i18n"), ("localization", "text_i18n"), ("parser-implementations", "parsing"), ("parsing", "parsing"),
    ("network-programming", "network"), ("web-programming", "network"),
    ("compression", "encoding"), ("encoding", "encoding"), ("multimedia::encoding", "encoding"),
    ("data-structures", "datastruct"), ("caching", "datastruct"), ("memory-management", "datastruct"),
    ("os", "os_hw"), ("hardware-support", "os_hw"), ("embedded", "os_hw"), ("os::windows-apis", "os_hw"),
]


def doc_paragraphs(text):
    out = []
    for blk in re.split(r"\n\s*\n", text):
        lines = [x.strip() for x in blk.splitlines()]
        if not lines:
            continue
        # skip numpy/google-style sections, doctest, code, field lists, rst directives
        if lines[0].startswith((">>>", "...", ".. ", "::", ":param", ":return", ":rtype", "@", "-----", "=====", "Parameters", "Returns", "Examples", "See Also", "Notes", "Raises", "Yields", "Attributes", "References")):
            continue
        lines = [l for l in lines if not l.startswith((">>>", "...", ".. ", ":param", ":return", ":rtype"))]
        p = h10c.clean_inline(" ".join(lines))
        p = re.sub(r":(?:func|class|meth|mod|attr|ref|exc|data|obj)\:`~?([^`]*)`", r"\1", p)
        p = re.sub(r"``([^`]*)``", r"\1", p)
        if 80 <= len(p) and h10c.ok_paragraph(p) and len(h10c.sentences(p)) >= 3:
            out.append(p)
    return out


def py_file_paras(path):
    try:
        src = open(path, encoding="utf8", errors="strict").read()
        tree = ast.parse(src)
    except Exception:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            ds = ast.get_docstring(node, clean=True)
            if ds:
                out.extend(doc_paragraphs(ds))
    return out


def collect_py(domain, roots, rng):
    files = []
    for base, name in roots:
        p = os.path.join(base, name)
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for f in glob.glob(p + "/**/*.py", recursive=True):
                fl = f.replace("\\", "/")
                if "/tests/" in fl or "/test_" in fl or "/_vendor" in fl or "/testing/" in fl or "/benchmarks/" in fl or "/externals/" in fl:
                    continue
                files.append(f)
    rng.shuffle(files)
    docs, seen = [], set()
    for f in files:
        got = 0
        paras = py_file_paras(f)
        rng.shuffle(paras)
        for p in paras:
            k = h10c.norm(p)[:70]
            if k in seen:
                continue
            seen.add(k)
            docs.append({"domain": domain, "src": os.path.relpath(f, HOME).replace("\\", "/"), "text": p})
            got += 1
            if got >= CAP_FILE:
                break
        if len(docs) >= CAP_DOMAIN * 3:
            break
    rng.shuffle(docs)
    return docs[:CAP_DOMAIN]


def project_docs(rng):
    docs, seen = [], set()
    files = sorted(glob.glob(ROOT + "/results/*.md")) + [ROOT + "/DESIGN.md"]
    for f in files:
        t = h10c.read(f)
        if not t:
            continue
        for head, paras in h10c.parse_md(t):
            for p in paras:
                if len(h10c.sentences(p)) >= 3 and len(p) >= 80:
                    k = h10c.norm(p)[:70]
                    if k not in seen:
                        seen.add(k)
                        docs.append({"domain": "project_research", "src": os.path.relpath(f, ROOT).replace("\\", "/"), "text": p})
    rng.shuffle(docs)
    return docs[:CAP_DOMAIN]


def crate_domain(toml):
    m = re.search(r"categories\s*=\s*\[(.*?)\]", toml, flags=re.S)
    if not m:
        return None
    cats = re.findall(r'"([^"]+)"', m.group(1))
    for want, dom in CRATE_CAT:
        if want in cats:
            return dom
    return None


def collect_crates(rng):
    base = glob.glob(HOME + "/.cargo/registry/src/*/")[0]
    by = collections.defaultdict(list)
    seen = set()
    for d in sorted(os.listdir(base)):
        cd = os.path.join(base, d)
        toml = h10c.read(os.path.join(cd, "Cargo.toml"))
        if not toml:
            continue
        dom = crate_domain(toml)
        if not dom:
            continue
        paras = []
        for rd in ("README.md", "readme.md", "README"):
            t = h10c.read(os.path.join(cd, rd))
            if t:
                for head, ps in h10c.parse_md(t):
                    paras.extend(ps)
                break
        lt = h10c.read(os.path.join(cd, "src", "lib.rs"))
        if lt:
            paras.extend(h10c.parse_comments(lt, "rs"))
        rng.shuffle(paras)
        got = 0
        for p in paras:
            if len(p) < 80 or len(h10c.sentences(p)) < 3:
                continue
            k = h10c.norm(p)[:70]
            if k in seen:
                continue
            seen.add(k)
            by[dom].append({"domain": dom, "src": "crate:" + d, "text": p})
            got += 1
            if got >= 15:
                break
    out = []
    for dom, ds in sorted(by.items()):
        rng.shuffle(ds)
        out.extend(ds[:CAP_DOMAIN])
    return out


def write(tag, docs):
    for i, d in enumerate(docs):
        d["id"] = i
    cnt = collections.Counter(d["domain"] for d in docs)
    meta = {"tag": tag, "n": len(docs), "per_domain": dict(cnt),
            "mean_chars": sum(len(d["text"]) for d in docs) / len(docs),
            "sha1": hashlib.sha1("\n".join(d["text"] for d in docs).encode()).hexdigest()}
    json.dump({"docs": docs, "meta": meta}, open(os.path.join(RES, "h20_corpus_%s.json" % tag), "w", encoding="utf8"))
    print(tag, json.dumps(meta))


def main():
    rng = random.Random(20260101)
    docs = []
    for dom, roots in PY9.items():
        d = collect_py(dom, roots, rng)
        print(dom, len(d), flush=True)
        docs.extend(d)
    pd = project_docs(rng)
    print("project_research", len(pd))
    docs.extend(pd)
    write("pydoc9", docs)
    cd = collect_crates(rng)
    write("crates8", cd)


if __name__ == "__main__":
    main()
