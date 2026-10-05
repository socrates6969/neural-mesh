"""h10: mine REAL human misspellings that occur in the real corpora (docs, headings) -> results/h10_real_typos.json
Heuristic (no dictionary on disk, so the bert-base-uncased WORD vocabulary of the on-disk MiniLM tokenizer is the 'valid word' list):
  candidate t: lowercase alphabetic token, len >= 6, written in lowercase in the source, NOT a whole-word entry of the BERT vocab,
  corpus frequency <= 2, and exactly one edit (substitute / delete / insert / transpose) away from a BERT-vocab word w whose corpus
  frequency >= 8, excluding pure inflection pairs (s/es/ed/ing/er/ly/d/y suffix differences) and hyphen/underscore fragments.
The surviving list is dumped for MANUAL REVIEW; an entry is kept only if a human reader (me) agrees it is a misspelling, listed in
KEEP_REVIEWED below after inspection."""
import json, os, re, sys, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
WORD = re.compile(r"[A-Za-z]+")
SUFF = ("s", "es", "ed", "ing", "er", "ly", "d", "y", "r", "al", "ion", "ers", "ness", "ment")


def bert_words():
    d = json.load(open(os.path.join(os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2"), "tokenizer.json"), encoding="utf8"))
    return {k for k in d["model"]["vocab"] if k.isalpha() and k.isascii() and len(k) >= 3}


def edits1(w):
    L = "abcdefghijklmnopqrstuvwxyz"
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in sp if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
    out |= {a + c + b[1:] for a, b in sp if b for c in L}
    out |= {a + c + b for a, b in sp for c in L}
    out.discard(w)
    return out


def inflect(t, w):
    for a, b in ((t, w), (w, t)):
        if b.startswith(a) and b[len(a):] in SUFF:
            return True
    return False


def mine(texts):
    cnt, cased = collections.Counter(), collections.Counter()
    for tx in texts:
        for m in WORD.findall(tx):
            cnt[m.lower()] += 1
            if m != m.lower():
                cased[m.lower()] += 1
    bw = bert_words()
    out = []
    for t, c in cnt.items():
        if c > 2 or len(t) < 6 or t in bw or cased[t] > 0:
            continue
        best = None
        for w in edits1(t):
            if w in bw and cnt.get(w, 0) >= 8 and not inflect(t, w):
                if best is None or cnt[w] > cnt[best]:
                    best = w
        if best:
            out.append((t, best, c, cnt[best]))
    return sorted(out, key=lambda x: -x[3])


if __name__ == "__main__":
    texts = []
    for tag in "AB":
        c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
        texts += [d["text"] for d in c["docs"]]
        texts += [q.get("orig", q["text"]) for q in c["queries"]]
    out = mine(texts)
    print(len(out))
    for t, w, c, cw in out[:400]:
        print(t, "->", w, c, cw)
