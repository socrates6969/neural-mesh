"""Retrieval over the prebrief corpus: BM25 (mesh defaults k1 .9, b .5), MiniLM-384 dense cosine, convex fusion,
prequential availability (only documents older than the query), and the anchor-based relevance label used offline."""
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict

import numpy as np

NM = os.path.join(os.path.expanduser("~"), "neural-mesh")
EMBED_EXE = os.path.join(NM, "target", "release", "embed_texts.exe")
WEIGHTS = os.path.join(NM, "data", "lifelong", "models", "minilm.f32")
TOKENIZER = os.path.join(NM, "data", "lifelong", "models", "tokenizer.json")
DIM = 384

STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its me my no not of on or our
please so that the their them then there these this to us was we were what when where which who why will with you your
should would could also all any each just more most other some such than too very via per use using used make sure""".split())
_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
_LONGHEX = re.compile(r"[0-9a-f]{12,}")


def tokens(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text):
        lw = w.lower()
        if len(lw) < 2 or lw in STOP or _LONGHEX.fullmatch(lw):  # long hex ids add noise (hebbwise-rgi.md: 114 hash features)
            continue
        out.append(lw)
        if "_" in w or re.search(r"[a-z][A-Z]", w):
            for part in re.split(r"_+", w):
                out.extend(p.lower() for p in (_CAMEL.findall(part) or [part]) if len(p) > 1 and p.lower() != lw)
    return out


# ---- anchors (offline relevance label, pre-registered) ----
_PATH = re.compile(r"(?:[A-Za-z]:)?[\w.\-/\\]*[\w\-]+\.(?:py|rs|ts|tsx|js|mjs|json|jsonl|md|toml|yaml|yml|ps1|sh|sql|tf|html|css|csv|txt|prisma|xlsx|pdf)\b")
_HASH = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")
_IDENT = re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+)\b")
_TICK = re.compile(r"`([^`\n]{3,80})`")


def anchors(text: str) -> set:
    a = set()
    for m in _PATH.findall(text):
        a.add("p:" + re.split(r"[/\\]", m)[-1].lower())
    for m in _HASH.findall(text):
        a.add("h:" + m[:7])
    for m in _IDENT.findall(text):
        if len(m) >= 6:
            a.add("i:" + m.lower())
    for m in _TICK.findall(text):
        a.add("t:" + m.strip().lower())
    return a


def _key(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:16]


class Index:
    def __init__(self, docs: list, cache_dir: str | None = None):
        self.docs = docs
        self.ts = np.array([d["ts"] for d in docs])
        self.tf = [Counter(tokens(d["text"])) for d in docs]
        self.dl = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t, n in c.items():
                self.post[t].append((i, n))
        self.anch = [anchors(d["text"]) for d in docs]
        self.agent = np.array([d.get("agent", "") for d in docs])
        self.session = np.array([d.get("session", "") for d in docs])
        self.cache_dir = cache_dir
        self.vec = None

    # -- dense --
    @staticmethod
    def dense_text(d: dict) -> str:
        body = d.get("report") or d["text"]
        return f"{d['title']}\n{body[:1200]}"

    def embed(self, texts: list):
        if not (os.path.exists(EMBED_EXE) and os.path.exists(WEIGHTS)):
            return None
        cache, kp, vp = {}, None, None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            kp, vp = os.path.join(self.cache_dir, "keys.json"), os.path.join(self.cache_dir, "vecs.f32")
            if os.path.exists(kp) and os.path.exists(vp):
                keys = json.load(open(kp))
                mat = np.fromfile(vp, dtype="<f4").reshape(-1, DIM)
                cache = {k: mat[r] for k, r in keys.items() if r < mat.shape[0]}
        missing = list({_key(t): t for t in texts if _key(t) not in cache}.items())
        if missing:
            tmp = os.path.join(self.cache_dir or os.getcwd(), f"pb_in_{os.getpid()}.json")
            out = tmp[:-5] + ".f32"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump([t for _, t in missing], f, ensure_ascii=False)
            r = subprocess.run([EMBED_EXE, WEIGHTS, TOKENIZER, tmp, out, "--threads", "4"], capture_output=True, text=True, timeout=3600)
            if r.returncode != 0:
                return None
            v = np.fromfile(out, dtype="<f4").reshape(-1, DIM)
            for (k, _), row in zip(missing, v):
                cache[k] = row
            for p in (tmp, out):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            if self.cache_dir:
                ks = list(cache)
                np.stack([cache[k] for k in ks]).astype("<f4").tofile(vp)
                with open(kp, "w") as f:
                    json.dump({k: i for i, k in enumerate(ks)}, f)
        m = np.stack([cache[_key(t)] for t in texts]).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        return m

    def ensure_vectors(self) -> bool:
        if self.vec is None:
            self.vec = self.embed([self.dense_text(d) for d in self.docs])
        return self.vec is not None

    # -- scoring --
    def bm25(self, qtoks: list, avail, k1: float = 0.9, b: float = 0.5):
        s = np.zeros(len(self.docs))
        n = int(avail.sum())
        if n == 0:
            return s
        avgdl = float(self.dl[avail].mean()) or 1.0
        for t, qn in Counter(qtoks).items():
            pl = [(i, f) for i, f in self.post.get(t, ()) if avail[i]]
            if not pl:
                continue
            idf = math.log(1 + (n - len(pl) + 0.5) / (len(pl) + 0.5))
            w = 1 + 0.3 * (qn - 1)
            for i, f in pl:
                s[i] += w * idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.dl[i] / avgdl))
        return s

    def available(self, before_ts=None, exclude_session=None):
        a = np.ones(len(self.docs), dtype=bool) if before_ts is None else self.ts < before_ts
        if exclude_session:
            a = a & (self.session != exclude_session)
        return a

    def search(self, query: str, role, avail, method: str = "P", alpha: float = 0.5, role_boost: float = 1.15, qvec=None):
        """Availability-masked scores (unavailable = -inf). method R=recency, B=BM25, P=BM25+dense fusion (+role boost)."""
        if method == "R":
            s = np.argsort(np.argsort(self.ts)).astype(float)
        else:
            s = self.bm25(tokens(query), avail)
            if method == "P":
                if qvec is None and self.ensure_vectors():
                    qvec = self.embed([query[:1500]])[0]
                if qvec is not None and self.vec is not None and avail.any():
                    d = self.vec @ qvec

                    def norm(x):
                        lo, hi = x[avail].min(), x[avail].max()
                        return (x - lo) / (hi - lo + 1e-9)
                    s = (1 - alpha) * norm(s) + alpha * norm(d)
                if role:
                    s = s * np.where(self.agent == role, role_boost, 1.0)
        s = s.astype(float)
        s[~avail] = -np.inf
        return s


def relevant(idx: Index, report: str, avail, max_df: int = 25, min_shared: int = 2, qa=None) -> set:
    qa = anchors(report) if qa is None else qa
    if not qa:
        return set()
    cand = np.flatnonzero(avail)
    df = Counter()
    for i in cand:
        for a in idx.anch[i] & qa:
            df[a] += 1
    keep = {a for a in qa if 0 < df[a] <= max_df}
    if not keep:
        return set()
    return {int(i) for i in cand if len(idx.anch[i] & keep) >= min_shared}