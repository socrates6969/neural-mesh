"""bench/frontend.py - the mesh retrieval front-end used by conditions B and D (moved verbatim from run.py:19-108).
Its file sha256 is the pre-registration key (PREREG.json "frontend_sha"): editing this file invalidates HELD/CONF runs.
Configuration is read from the environment at import time: MESH_THRESH (default 30.0), MESH_V2=1 (identifier-aware
lookup), MESH_FRONTEND=hook (the real UserPromptSubmit hook instead of BM25)."""
import json, math, os, pathlib, re, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
NM = HERE.parent
THRESH = float(os.environ.get("MESH_THRESH", "30.0"))

# ------------------------------------------------------------------ retrieval (BM25, stdlib)
def build_corpus():
    files = [NM / "DESIGN.md", NM / "RD-LOG.md", NM / "RD-LOG-B.md"] + sorted((NM / "src").glob("*.rs"))
    chunks = []
    for f in files:
        lines = f.read_text(encoding="utf8", errors="replace").splitlines()
        for i in range(0, len(lines), 25):
            seg = lines[max(0, i - 3): i + 28]
            chunks.append((f"{f.relative_to(NM).as_posix()}:{i+1}", "\n".join(seg)))
    return chunks
TOK = re.compile(r"[A-Za-z0-9_]+")
def toks(s): return [t.lower() for t in TOK.findall(s)]
class BM25:
    def __init__(self, chunks, k1=1.5, b=.75):
        self.chunks = chunks; self.k1 = k1; self.b = b
        self.tf = [];
        df = {}
        for _, t in chunks:
            c = {}
            for w in toks(t): c[w] = c.get(w, 0) + 1
            self.tf.append(c)
            for w in c: df[w] = df.get(w, 0) + 1
        self.N = len(chunks); self.df = df
        self.len = [sum(c.values()) for c in self.tf]; self.avg = sum(self.len) / self.N
    def search(self, q, k=3):
        qs = set(toks(q)); sc = []
        for i, c in enumerate(self.tf):
            s = 0.0
            for w in qs:
                f = c.get(w)
                if not f: continue
                idf = math.log(1 + (self.N - self.df[w] + .5) / (self.df[w] + .5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            sc.append((s, i))
        sc.sort(reverse=True)
        return [(s, self.chunks[i][0], self.chunks[i][1]) for s, i in sc[:k]]
_bm = None
def retrieve(q):
    global _bm
    if _bm is None: _bm = BM25(build_corpus())
    return _bm.search(q, 3)

V2 = os.environ.get("MESH_V2", "0") == "1"
IDENT = re.compile(r"`([A-Za-z_][A-Za-z0-9_ ]*[A-Za-z0-9_])`")
PATH = re.compile(r"src/[A-Za-z0-9_/]+[.]rs")
def retrieve_v2(q):
    """identifier-aware lookup: if the query names code identifiers (`NAME`) or a src/ path, score only those tokens
    inside the named file. Returns (hits, structured?)."""
    global _bm
    if _bm is None: _bm = BM25(build_corpus())
    ids = [t for m in IDENT.findall(q) for t in toks(m)]
    paths = PATH.findall(q)
    if not ids and not paths: return _bm.search(q, 3), False
    sub = [i for i, (src, _) in enumerate(_bm.chunks) if not paths or src.startswith(paths[0] + ":")]
    qs = set(ids); sc = []
    for i in sub:
        c = _bm.tf[i]; sv = 0.0
        for w in qs:
            f = c.get(w)
            if f: sv += math.log(1 + (_bm.N - _bm.df[w] + .5) / (_bm.df[w] + .5)) * f * 2.5 / (f + 1.5 * (1 - .75 + .75 * _bm.len[i] / _bm.avg))
        sc.append((sv + 5, i))   # +5: the path filter itself is evidence
    sc.sort(reverse=True)
    return [(s_, _bm.chunks[i][0], _bm.chunks[i][1]) for s_, i in sc[:3]], True
def query_of(prompt):
    # drop the answer-format boilerplate so it cannot match anything
    return re.sub(r"\n\nGive the final answer.*|\nReturn ONLY the code.*", "", prompt, flags=re.S)
HOOK_FE = os.environ.get("MESH_FRONTEND", "") == "hook"     # B = the real UserPromptSubmit hook (hooks/recall_hook.py), run as a subprocess
_hook_data = None
def hook_frontend(prompt):
    """The exact hook: same script, same gate + recall + floor + scrub, fed the hook JSON with cwd = this repo (the prompts are about it).
    Reads a FROZEN copy of the memory ledger (so a nightly ingest cannot change the corpus mid-run) and never writes retrieval events."""
    global _hook_data
    if _hook_data is None:
        import shutil, tempfile
        _hook_data = tempfile.mkdtemp(prefix="bench-hook-"); (pathlib.Path(_hook_data) / "memory").mkdir()
        shutil.copy(NM / "data" / "lifelong" / "memory" / "items.jsonl", pathlib.Path(_hook_data) / "memory" / "items.jsonl")
    env = dict(os.environ, INGEST_DATA=_hook_data)
    r = subprocess.run([sys.executable, "-S", str(NM / "hooks" / "recall_hook.py")], input=json.dumps(dict(prompt=prompt, cwd=str(NM), session_id="bench", hook_event_name="UserPromptSubmit")),
                       capture_output=True, text=True, encoding="utf8", errors="replace", env=env, timeout=15)
    out = r.stdout.strip()
    if not out: return prompt, {"retrieved": False, "hook": True}
    return f"{out}" + chr(10)*2 + "---" + chr(10) + prompt, {"retrieved": True, "hook": True, "hook_chars": len(out)}
def frontend_prompt(prompt):
    if HOOK_FE: return hook_frontend(prompt)
    if V2: hits, structured = retrieve_v2(query_of(prompt))
    else: hits, structured = retrieve(query_of(prompt)), False
    top = hits[0][0]
    if top < (8.0 if structured else THRESH): return prompt, {"retrieved": False, "top": round(top, 2)}
    ctx = "\n\n".join(f"[{src}]\n{txt[:2200]}" for _, src, txt in hits[:2])
    return f"Context retrieved from the project repository (may or may not be relevant):\n{ctx}\n\n---\n{prompt}", {"retrieved": True, "top": round(top, 2)}


# ------------------------------------------------------------------ provenance helpers (additive; not used by the retrieval path)
def corpus_files():
    return [NM / "DESIGN.md", NM / "RD-LOG.md", NM / "RD-LOG-B.md"] + sorted((NM / "src").glob("*.rs"))
def corpus_sha():
    import hashlib
    h = hashlib.sha256()
    for f in corpus_files():
        h.update(f.relative_to(NM).as_posix().encode() + b"|" + f.read_bytes().replace(b"\r\n", b"\n") + b"|")
    return h.hexdigest()
def frontend_sha():
    import hashlib
    return hashlib.sha256(pathlib.Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # CRLF-independent (core.autocrlf)
def frontend_env():
    """the env knobs that select the front-end variant (recorded in manifests, pinned in PREREG.json)"""
    return {k: os.environ.get(k, "") for k in ("MESH_V2", "MESH_THRESH", "MESH_FRONTEND")}
