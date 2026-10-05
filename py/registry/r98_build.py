"""R98 builder: candidate-selection prompts for a MODEL-IN-THE-LOOP test of trimmed catalog lines (R73's open caveat: 'whether a model can still choose from a trimmed line is NOT measured').
For every query of Q + Q2 + Q3 (R31/R32/R74; dedup) the top-5 default-hybrid(.7) registry entries are listed in retrieval order in four line formats: full / <=15 words / <=8 words / name only.
Writes sel_<arm>.txt (prompts) + sel_key.json (acceptable names, candidate names, ranks) to the scratchpad dir given as argv[1]."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_eval import *
from q3 import Q3
out = sys.argv[1]
ents, texts, E = load(); bm = BM25(texts)
allq = []; seen = set()
for q, acc in list(Q) + list(globals().get("Q2", [])) + list(Q3):
    if q not in seen: seen.add(q); allq.append((q, acc))
QE = embed([q for q, _ in allq]); key = []
def trim(d, n): w = d.split(); return " ".join(w[:n]) + (" ..." if len(w) > n else "")
FORMATS = {"full": lambda e: f"{e['kind']} {e['name']}: {e['desc']}", "w15": lambda e: f"{e['kind']} {e['name']}: {trim(e['desc'], 15)}", "w8": lambda e: f"{e['kind']} {e['name']}: {trim(e['desc'], 8)}", "name": lambda e: f"{e['kind']} {e['name']}"}
tops = []
for i, (q, acc) in enumerate(allq):
    sc = .7 * z(E @ QE[i]) + .3 * z(bm.score(q)); top = list(np.argsort(-sc)[:5]); tops.append(top)
    key.append(dict(q=q, acc=sorted(acc), cands=[ents[j]["name"] for j in top], ok=[ents[j]["name"].lower() in {a.lower() for a in acc} for j in top]))
HEAD = ("You are choosing a capability for each request. For every numbered request below, 5 candidate capabilities are listed (kind name: description). "
        "Pick the ONE candidate (1-5) that best serves the request, or 0 if none of them is appropriate. Answer from the text only. "
        "Write your answers to the output file named in your instructions, one line per request, exactly in the form `Q7: 3` (no commentary). Do not read any other file.\n\n")
for arm, fmt in FORMATS.items():
    lines = [HEAD]
    for i, (q, acc) in enumerate(allq):
        lines.append(f"Q{i}: {q}")
        for r, j in enumerate(tops[i]): lines.append(f"  {r+1}) {fmt(ents[j])}")
        lines.append("")
    txt = "\n".join(lines); open(os.path.join(out, f"sel_{arm}.txt"), "w", encoding="utf-8").write(txt)
    import tiktoken; print(arm, "prompt chars", len(txt), "cl100k tokens (proxy)", len(tiktoken.get_encoding("cl100k_base").encode(txt)))
json.dump(key, open(os.path.join(out, "sel_key.json"), "w"))
print("queries", len(allq), "with an acceptable candidate in top-5:", sum(any(k["ok"]) for k in key), " top-1 acceptable:", sum(k["ok"][0] for k in key))
