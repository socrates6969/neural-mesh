"""B1 battery generator: 25 cognitive sub-skill PROXIES + knowledge stores. Deterministic per (world, seed).
Writes results/b1_items.json, results/b1_texts.json, results/b1_emb.f32 (MiniLM, via onnxruntime; model in ~/.ruvector).
EVERYTHING here is a SYNTHETIC PROXY authored by the R&D director. See RD-LOG.md for the scope statement."""
import itertools, json, os, random, sys
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_domain import DOMAINS
import data_verbal as V

SKILLS = ["vocab","similarities","general_info","comprehension","verbal_analogies","sentence_completion",
          "pattern_recognition","matrix_reasoning","inductive","deductive","series","classification",
          "digit_span_fwd","digit_span_back","letter_number_seq","mental_arithmetic",
          "symbol_coding","symbol_search","visual_cancellation",
          "block_design","visual_puzzles","spatial_rotation","paper_folding",
          "numerical_ops","quant_problem_solving"]
FAMILY = {s: f for f, ss in {"verbal": SKILLS[0:6], "fluid": SKILLS[6:12], "wm": SKILLS[12:16], "speed": SKILLS[16:19], "spatial": SKILLS[19:23], "quant": SKILLS[23:25]}.items() for s in ss}
SEMANTIC = {"vocab","similarities","verbal_analogies","sentence_completion","classification"}
KNOW = {"general_info","comprehension"}
N_PER = 6
CHAT = ["Quick question: ", "Hey, could you help me with this? ", "For a puzzle I'm doing: ", "Please answer carefully. ", "Not sure about this one. ", ""]
OUTRO = [" Thanks!", " Show your work briefly.", "", " (no calculator please)", " I need it for homework."]

def wrap(r, w, s):
    return s if w == "A" else r.choice(CHAT) + s + r.choice(OUTRO)

def lst(a): return "[" + ",".join(str(x) for x in a) + "]"
def bad_int(r, v):
    c = [v + 1, v - 1, v + 10, v - 10, v * 2 if v else 3]
    return r.choice([x for x in c if x != v])
def bad_list(r, a):
    a = list(a)
    if len(a) >= 2 and r.random() < .5:
        i = r.randrange(len(a) - 1); b = a[:]; b[i], b[i + 1] = b[i + 1], b[i]
        if b != a: return b
    i = r.randrange(len(a)); b = a[:]; b[i] = b[i] + r.choice([1, -1, 2]); return b

def gen_proc(skill, r, w):
    """returns dict(text, tool, spec, ans) ; ans is the tool-output string"""
    if skill == "mental_arithmetic":
        a, b, c = r.randint(2, 40 if w == "A" else 99), r.randint(2, 30 if w == "A" else 99), r.randint(2, 9)
        e = r.choice([f"{a}+{b}*{c}", f"({a}+{b})*{c}", f"{a}*{c}-{b}", f"{a}*{b}-{c}"])
        v = eval(e)
        return dict(text=wrap(r, w, f"What is {e.replace('*','x') if w=='A' else e.replace('*',' times ')}?"), tool="calc", spec=e, ans=str(v), bad=str(bad_int(r, v)))
    if skill == "numerical_ops":
        p, q, s, t = [r.randint(1, 9) for _ in range(4)]
        e = r.choice([f"{p}/{q}+{s}/{t}", f"{p}/{q}*{s}/{t}", f"{p}^{r.randint(2,4)}-{q}", f"{p}/{q}-{s}/{t}"])
        v = Fraction(eval(e.replace("^", "**").replace("/", "/").replace("1/", "Fraction(1)/") if False else "0")) if False else None
        ee = e.replace("^", "**")
        import re
        ee = re.sub(r"(\d+)", r"Fraction(\1)", ee)
        v = eval(ee)
        v = Fraction(v)
        ans = str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
        bv = v + 1; bad = str(bv.numerator) if bv.denominator == 1 else f"{bv.numerator}/{bv.denominator}"
        return dict(text=wrap(r, w, f"Compute {e} exactly."), tool="calc", spec=e, ans=ans, bad=bad)
    if skill == "quant_problem_solving":
        k = r.randrange(4)
        if k == 0:
            sp, h = r.randint(20, 90), r.randint(2, 9)
            return dict(text=wrap(r, w, f"A train travels at {sp} km/h for {h} hours. How many km does it cover?"), tool="calc", spec=f"{sp}*{h}", ans=str(sp * h), bad=str(sp * h + sp))
        if k == 1:
            pr, d = r.choice([40, 60, 80, 120, 200]), r.choice([10, 20, 25, 50])
            v = Fraction(pr * (100 - d), 100)
            ans = str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
            return dict(text=wrap(r, w, f"An item costs {pr} dollars and is discounted by {d} percent. What is the sale price in dollars?"), tool="calc", spec=f"{pr}*(100-{d})/100", ans=ans, bad=str(pr - d))
        if k == 2:
            n, c = r.randint(3, 12), r.randint(2, 15)
            return dict(text=wrap(r, w, f"{n} notebooks cost {c} dollars each and a pen costs 3 dollars. What is the total for the notebooks and one pen?"), tool="calc", spec=f"{n}*{c}+3", ans=str(n * c + 3), bad=str(n * c))
        km = r.randint(2, 50)
        return dict(text=wrap(r, w, f"How many metres are in {km} kilometres?"), tool="units", spec=f"conv {km} km -> m", ans=str(km * 1000), bad=str(km * 100))
    if skill in ("digit_span_fwd", "digit_span_back"):
        n = r.randint(4, 7) if w == "A" else r.randint(6, 9)
        a = [r.randint(0, 9) for _ in range(n)]
        if skill == "digit_span_fwd":
            return dict(text=wrap(r, w, "Repeat these digits in the same order: " + " ".join(map(str, a))), tool="sandbox", spec=f"let a={lst(a)}; return a;", ans=lst(a), bad=lst(bad_list(r, a)))
        o = a[::-1]
        return dict(text=wrap(r, w, "Say these digits backwards: " + " ".join(map(str, a))), tool="sandbox", spec=f"let a={lst(a)}; let o=[]; let i=len(a)-1; while i>=0 {{ o=push(o,a[i]); i=i-1; }} return o;", ans=lst(o), bad=lst(bad_list(r, o)))
    if skill == "letter_number_seq":
        n = r.randint(4, 6) if w == "A" else r.randint(6, 8)
        toks = [r.choice("0123456789") if r.random() < .5 else r.choice("ABCDEFGHJK") for _ in range(n)]
        enc = [int(t) if t.isdigit() else 100 + ord(t) for t in toks]
        prog = f"let a={lst(enc)}; let n=len(a); let i=0; while i<n {{ let j=0; while j<n-1-i {{ if a[j]>a[j+1] {{ let t=a[j]; a[j]=a[j+1]; a[j+1]=t; }} j=j+1; }} i=i+1; }} return a;"
        o = sorted(enc)
        return dict(text=wrap(r, w, "Reorder: digits first in ascending order, then letters alphabetically: " + " ".join(toks)), tool="sandbox", spec=prog, ans=lst(o), bad=lst(bad_list(r, o)))
    if skill == "symbol_coding":
        key = r.sample(range(1, 10), 9)  # digit d -> code key[d-1]
        n = 8 if w == "A" else 14
        s = [r.randint(1, 9) for _ in range(n)]
        o = [key[d - 1] for d in s]
        prog = f"let k={lst(key)}; let s={lst(s)}; let o=[]; let i=0; while i<len(s) {{ o=push(o,k[s[i]-1]); i=i+1; }} return o;"
        return dict(text=wrap(r, w, f"Using key (digit n -> symbol code at position n) {lst(key)}, encode the digits {lst(s)}."), tool="sandbox", spec=prog, ans=lst(o), bad=lst(bad_list(r, o)))
    if skill == "symbol_search":
        tg = r.sample(range(1, 9), 2); grp = r.sample(range(1, 9), 5 if w == "A" else 7)
        v = 1 if any(t in grp for t in tg) else 0
        prog = f"let t={lst(tg)}; let g={lst(grp)}; let f=0; let i=0; while i<len(t) {{ let j=0; while j<len(g) {{ if t[i]==g[j] {{ f=1; }} j=j+1; }} i=i+1; }} return f;"
        return dict(text=wrap(r, w, f"Is any of the target symbols {lst(tg)} present in the group {lst(grp)}? Answer 1 for yes, 0 for no."), tool="sandbox", spec=prog, ans=str(v), bad=str(1 - v))
    if skill == "visual_cancellation":
        n = 20 if w == "A" else 40
        g = [r.randint(1, 5) for _ in range(n)]; t = r.randint(1, 5); v = g.count(t)
        prog = f"let g={lst(g)}; let c=0; let i=0; while i<len(g) {{ if g[i]=={t} {{ c=c+1; }} i=i+1; }} return c;"
        return dict(text=wrap(r, w, f"Count how many times mark {t} appears in the array {lst(g)}."), tool="sandbox", spec=prog, ans=str(v), bad=str(v + 1 if v == 0 else v - 1))
    if skill == "pattern_recognition":
        p = r.randint(2, 4); base = r.sample(range(1, 9), p); reps = r.randint(2, 3)
        a = (base * (reps + 1))[: p * reps + r.randint(1, p - 1)]
        v = base[len(a) % p]
        return dict(text=wrap(r, w, f"What comes next in the repeating pattern {lst(a)}?"), tool="sandbox", spec=f"let a={lst(a)}; return a[len(a)-{p}];", ans=str(v), bad=str(bad_int(r, v)))
    if skill == "matrix_reasoning":
        rule = r.choice(["add", "sub", "mul"]) if w == "A" else r.choice(["add", "sub", "mul", "col"])
        rows = []
        for _ in range(3):
            a, b = r.randint(2, 9), r.randint(1, 9)
            if rule == "add": rows.append([a, b, a + b])
            elif rule == "sub": rows.append([a + b, b, a])
            elif rule == "mul": rows.append([a, b, a * b])
            else: rows.append([a, b, a + b])
        if rule == "col":
            m = [[r.randint(1, 5) for _ in range(3)] for _ in range(2)]; m.append([m[0][c] + m[1][c] for c in range(3)]); rows = m
            hid = rows[2][2]; spec = f"{rows[0][2]}+{rows[1][2]}"
        else:
            hid = rows[2][2]
            spec = {"add": f"{rows[2][0]}+{rows[2][1]}", "sub": f"{rows[2][0]}-{rows[2][1]}", "mul": f"{rows[2][0]}*{rows[2][1]}"}[rule]
        cells = [str(rows[i][j]) if (i, j) != (2, 2) else "?" for i in range(3) for j in range(3)]
        return dict(text=wrap(r, w, "Find the missing cell '?' in the 3x3 grid (row by row): " + " ".join(cells)), tool="calc", spec=spec, ans=str(hid), bad=str(hid + 1))
    if skill == "inductive":
        a, b = r.randint(2, 6), r.randint(0, 9); x0 = r.randint(1, 4)
        q = x0 + 4 + r.randint(0, 3)
        ex = ", ".join(f"f({x})={a*x+b}" for x in range(x0, x0 + 3))
        return dict(text=wrap(r, w, f"A rule gives {ex}. What is f({q})?"), tool="calc", spec=f"{a}*{q}+{b}", ans=str(a * q + b), bad=str(a * q + b + a))
    if skill == "series":
        k = r.randrange(4 if w == "A" else 5)
        if k == 0:
            d, s0 = r.randint(2, 9), r.randint(1, 20); a = [s0 + d * i for i in range(5)]; v = s0 + 5 * d; prog = f"let a={lst(a)}; return a[4]+{d};"
        elif k == 1:
            q, s0 = r.randint(2, 3), r.randint(1, 4); a = [s0 * q ** i for i in range(5)]; v = s0 * q ** 5; prog = f"let a={lst(a)}; return a[4]*{q};"
        elif k == 2:
            a = [(i + r.randint(1, 3)) ** 2 for i in range(0)]; o = r.randint(1, 4); a = [(o + i) ** 2 for i in range(5)]; v = (o + 5) ** 2; prog = f"return {o+5}*{o+5};"
        elif k == 3:
            x, y = r.randint(1, 4), r.randint(1, 4); a = [x, y]
            for _ in range(4): a.append(a[-1] + a[-2])
            v = a[-1] + a[-2]; prog = f"let a={lst(a)}; return a[4]+a[5];"
        else:
            s0, d1, d2 = r.randint(1, 9), r.randint(1, 5), r.randint(1, 5); a = [s0]
            for i in range(5): a.append(a[-1] + (d1 if i % 2 == 0 else d2))
            v = a[-1] + d1; prog = f"let a={lst(a)}; return a[5]+{d1};"
        return dict(text=wrap(r, w, f"Find the next number in the series {', '.join(map(str, a))}, ..."), tool="sandbox", spec=prog, ans=str(v), bad=str(bad_int(r, v)))
    if skill == "block_design":
        perm = [2, 0, 3, 1]  # one 90-degree clockwise rotation of a flattened 2x2 block
        blk = r.sample(range(1, 7), 4); k = r.randrange(4)
        tgt = blk
        for _ in range(k): tgt = [tgt[p] for p in perm]
        prog = f"let b={lst(blk)}; let t={lst(tgt)}; let p=[2,0,3,1]; let k=0; let found=0-1; while k<4 {{ let ok=1; let i=0; while i<4 {{ if b[i]!=t[i] {{ ok=0; }} i=i+1; }} if ok==1 && found<0 {{ found=k; }} let nb=[]; i=0; while i<4 {{ nb=push(nb,b[p[i]]); i=i+1; }} b=nb; k=k+1; }} return found;"
        # ambiguity: if the block has symmetry the first match is returned; recompute truth
        bb = blk; found = -1
        for kk in range(4):
            if bb == tgt and found < 0: found = kk
            bb = [bb[p] for p in perm]
        return dict(text=wrap(r, w, f"A 2x2 block with cells {lst(blk)} must match the target {lst(tgt)}. How many 90-degree clockwise rotations (0-3) are needed? Cells are read row by row; one rotation maps cell i to old cell [2,0,3,1][i]."), tool="sandbox", spec=prog, ans=str(found), bad=str((found + 1) % 4))
    if skill == "visual_puzzles":
        half = [r.randint(0, 1) for _ in range(3)]  # each row is mirror-symmetric: [a,b,a]
        g = []
        for i in range(3): g += [half[i], r.randint(0, 1), half[i]]
        m = r.choice([0, 2, 3, 5, 6, 8]); mir = {0: 2, 2: 0, 3: 5, 5: 3, 6: 8, 8: 6}[m]; v = g[mir]
        gg = g[:]; gg[m] = 9
        return dict(text=wrap(r, w, f"The 3x3 figure {lst(gg)} (row by row) is left-right symmetric; the cell holding 9 is hidden. What value is hidden?"), tool="sandbox", spec=f"let g={lst(gg)}; let m={{0:0}}; return 0;", ans=str(v), bad=str(1 - v), _special=("mirror", gg, m, mir))
    if skill == "spatial_rotation":
        x, y = r.randint(-6, 6), r.randint(-6, 6); k = r.randint(1, 3)
        px, py = x, y
        for _ in range(k): px, py = -py, px
        prog = f"let x={x}; let y={y}; let k=0; while k<{k} {{ let t=x; x=0-y; y=t; k=k+1; }} return [x,y];"
        return dict(text=wrap(r, w, f"The point ({x},{y}) is rotated {90*k} degrees counter-clockwise about the origin. Give its new coordinates."), tool="sandbox", spec=prog, ans=lst([px, py]), bad=lst([py, px] if (py, px) != (px, py) else [px + 1, py]))
    if skill == "paper_folding":
        W = 6 if w == "B" else 4
        rr, c = r.randint(0, W - 1), r.randint(0, W // 2 - 1)
        ans = sorted([c, W - 1 - c])
        prog = f"let c={c}; let m={W-1}-c; if c<m {{ return [c,m]; }} else {{ return [m,c]; }}"
        return dict(text=wrap(r, w, f"A {W}-wide square paper is folded once left-over-right along its vertical midline, and a hole is punched in column {c} (0-based) of the folded layers. After unfolding, which columns have holes? Give them in ascending order."), tool="sandbox", spec=prog, ans=lst(ans), bad=lst([c, c + 1]))
    raise KeyError(skill)

def gen_deductive(r, w):
    n = 4 if w == "A" else 5
    names = r.sample(list("abcdefgh"), n); perm = r.sample(range(1, n + 1), n); pos = dict(zip(names, perm))
    while True:
        cl = []
        for _ in range(n + 1 if w == "A" else n + 2):
            x, y = r.sample(names, 2)
            cl.append((x, y) if pos[x] < pos[y] else (y, x))
        cl = list(dict.fromkeys(cl))
        sols = []
        for p in itertools.permutations(range(1, n + 1)):
            d = dict(zip(names, p))
            if all(d[a] < d[b] for a, b in cl): sols.append(d)
            if len(sols) > 1: break
        if len(sols) == 1: break
    spec = "; ".join(f"{x} in 1..{n}" for x in names) + "; " + "; ".join(f"{a}<{b}" for a, b in cl) + "; " + "; ".join(f"{x}!={y}" for x, y in itertools.combinations(names, 2))
    first = [k for k, v in pos.items() if v == 1][0]
    ans = ",".join(f"{k}={pos[k]}" for k in names)
    bp = dict(pos); a1, a2 = names[0], names[1]; bp[a1], bp[a2] = bp[a2], bp[a1]
    clues = "; ".join(f"{a} is ahead of {b}" for a, b in cl)
    return dict(text=wrap(r, w, f"Runners {', '.join(names)} each finish in a distinct place 1..{n}. Clues: {clues}. Give each runner's place."), tool="csp", spec=spec, ans=ans, bad=",".join(f"{k}={bp[k]}" for k in names))

def gen_semantic(skill, r, w, texts):
    vocab = V.VOCAB_A if w == "A" else V.VOCAB_B; ont = V.ONT_A if w == "A" else V.ONT_B
    rel = V.REL_A if w == "A" else V.REL_B; cloze = V.CLOZE_A if w == "A" else V.CLOZE_B
    if skill == "vocab":
        wd, syn, ant = r.choice(vocab)
        others = [x[1] for x in vocab if x[0] != wd]
        opts = [syn, ant] + r.sample(others, 2); r.shuffle(opts)
        texts.update([wd] + opts)
        return dict(text=wrap(r, w, f"Which word is closest in meaning to '{wd}'? Options: {', '.join(opts)}."), target=wd, opts=opts, ans=opts.index(syn))
    if skill == "similarities":
        c1, c2, c3 = r.sample(list(ont), 3)
        x, y = r.sample(ont[c1], 2); opts = [c1, c2, c3]; r.shuffle(opts)
        texts.update([x, y] + opts)
        return dict(text=wrap(r, w, f"In what way are {x} and {y} alike? Options: {', '.join(opts)}."), pair=[x, y], opts=opts, ans=opts.index(c1))
    if skill == "classification":
        c1, c2 = r.sample(list(ont), 2); ms = r.sample(ont[c1], 3); odd = r.choice(ont[c2]); opts = ms + [odd]; r.shuffle(opts)
        texts.update(opts)
        return dict(text=wrap(r, w, f"Which one does not belong? {', '.join(opts)}."), opts=opts, ans=opts.index(odd))
    if skill == "verbal_analogies":
        rn = r.choice(list(rel)); pairs = r.sample(rel[rn], 4)
        (a, b), (c, d) = pairs[0], pairs[1]; opts = [d] + [p[1] for p in pairs[2:]] + [r.choice([p[1] for p in rel[rn] if p[1] not in (d,) + tuple(q[1] for q in pairs[2:])])]
        r.shuffle(opts); texts.update([a, b, c] + opts)
        return dict(text=wrap(r, w, f"{a} is to {b} as {c} is to ? Options: {', '.join(opts)}."), abc=[a, b, c], opts=opts, ans=opts.index(d))
    if skill == "sentence_completion":
        st, an, ds = r.choice(cloze); opts = [an] + ds; r.shuffle(opts)
        texts.add(st); texts.update(opts); texts.update(st.replace("___", o) for o in opts)
        return dict(text=wrap(r, w, f"Choose the word that best fills the blank: {st} Options: {', '.join(opts)}."), stem=st, opts=opts, ans=opts.index(an))
    raise KeyError(skill)

def gen_comprehension(r, w):
    names = ["Mia", "Omar", "Lena", "Ravi", "Sofia", "Kenji", "Aisha", "Tomas"]; places = ["the harbour", "the library", "the market", "the museum", "the station", "the park", "the bakery", "the clinic"]
    things = ["a red umbrella", "an old map", "a loaf of bread", "a silver key", "two tickets", "a heavy parcel", "a blue notebook", "a small lantern"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    k = 3 if w == "A" else 5
    ns = r.sample(names, k); sents = []; facts = []
    for n in ns:
        p, t, d = r.choice(places), r.choice(things), r.choice(days)
        sents.append(f"On {d}, {n} went to {p} and picked up {t}."); facts.append((n, p, t, d))
    i = r.randrange(k); n, p, t, d = facts[i]
    q, a = r.choice([(f"Where did {n} go?", p), (f"What did {n} pick up?", t), (f"On which day did {n} go out?", d)])
    qq = q if w == "A" else q.replace("Where did", "To which place did").replace("What did", "Which object did") + " (from the passage)"
    return dict(text=wrap(r, w, qq), sents=sents, passage=" ".join(sents), gold=i, answer=a)

def gen_world_seed(w, seed, texts):
    out = []
    for si, sk in enumerate(SKILLS):
        r = random.Random(seed * 100003 + si * 997 + (1 if w == "B" else 0))
        for _ in range(N_PER):
            if sk in SEMANTIC: it = gen_semantic(sk, r, w, texts)
            elif sk == "comprehension": it = gen_comprehension(r, w)
            elif sk == "general_info": it = None
            elif sk == "deductive": it = gen_deductive(r, w)
            else: it = gen_proc(sk, r, w)
            if it is None: continue
            if it.get("_special"):
                kind, gg, m, mir = it.pop("_special")
                mp = {0: 2, 2: 0, 3: 5, 5: 3, 6: 8, 8: 6}
                it["spec"] = f"let g={lst(gg)}; let m={m}; if m==0 {{ return g[2]; }} if m==2 {{ return g[0]; }} if m==3 {{ return g[5]; }} if m==5 {{ return g[3]; }} if m==6 {{ return g[8]; }} return g[6];"
            it["skill"] = sk; it["fam"] = FAMILY[sk]
            if "tool" in it: it["route"] = it["tool"]
            elif sk in SEMANTIC: it["route"] = "semantic"
            else: it["route"] = "knowledge"
            out.append(it)
    return out

def build_general():
    docs, qs = [], []
    for c, cap in V.CAPITALS:
        docs.append(f"{cap.title()} is the capital city of {c.title()}."); qs.append((len(docs) - 1, f"What is the capital of {c.title()}?", f"Of which country is {cap.title()} the capital?", cap.title()))
    for el, sy in V.ELEMENTS:
        docs.append(f"The chemical symbol for {el} is {sy}."); qs.append((len(docs) - 1, f"What is the chemical symbol for {el}?", f"Which element is represented by the symbol {sy}?", sy))
    for au, bk in V.AUTHORS:
        docs.append(f"{au} wrote {bk}."); qs.append((len(docs) - 1, f"Who wrote {bk}?", f"{au} is the author of which book?", au))
    for p, th in V.INVENT:
        docs.append(f"{p} is credited with {th}."); qs.append((len(docs) - 1, f"What is {p} credited with?", f"Who is credited with {th}?", p))
    for nm, loc, kind in V.GEO:
        docs.append(f"{nm.capitalize() if nm[0]!='M' and nm[0]!='K' and nm[0]!='L' else nm} is a {kind} in {loc}."); qs.append((len(docs) - 1, f"Where is {nm}?", f"Name a {kind} located in {loc}.", loc))
    return docs, qs

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
    texts = set()
    gdocs, gq = build_general()
    ddocs, dq = [], []
    for dom, es in DOMAINS.items():
        for d, q1, q2 in es:
            ddocs.append((dom, d)); dq.append((len(ddocs) - 1, dom, q1, q2))
    texts.update(gdocs); texts.update(x[1] for x in gq); texts.update(x[2] for x in gq)
    texts.update(d for _, d in ddocs); texts.update(x[2] for x in dq); texts.update(x[3] for x in dq)
    items = {}
    for w in "AB":
        for seed in range(1, 91):
            items[f"{w}{seed}"] = gen_world_seed(w, seed, texts)
            for it in items[f"{w}{seed}"]:
                if it["skill"] == "comprehension": texts.update(it["sents"])
                texts.add(it["text"])
    stores = dict(general=dict(docs=gdocs, qs=gq), domain=dict(docs=[d for _, d in ddocs], dom=[a for a, _ in ddocs], qs=dq))
    tl = sorted(texts)
    json.dump(dict(items=items, stores=stores), open(os.path.join(out_dir, "b1_items.json"), "w"))
    json.dump(tl, open(os.path.join(out_dir, "b1_texts.json"), "w"))
    print("items", sum(len(v) for v in items.values()), "texts", len(tl))

if __name__ == "__main__":
    main()
