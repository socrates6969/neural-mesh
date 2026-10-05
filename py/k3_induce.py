"""k3_induce (v3): generic prose template induction for 'flattened table' prose. Developed ONLY against DEV-split renders of families F1-F3 (regexes disabled
during tuning: KNOW_IND). NOTE (disclosure): H-family numbers on the dev split were printed during development, so the H families are design-adjacent, not blind.
No family-specific regex: sentences in a block are clustered by shared template words (document frequency >= tmpl_df); remaining word runs are row-label / column-header
slots; roles are decided by which slot varies slowly along the (row-major) sentence sequence."""
import re
from collections import Counter
from k3_data import norm_key, core_key

SENT_SPLIT = re.compile(r'(?<=[A-Za-z0-9\)\]\%])\.\s+(?=[A-Z])')
TOK = re.compile(r'\S+')
STOP = {'for', 'the', 'is', 'are', 'was', 'of', 'at', 'in', 'on', 'a', 'an', 'and', 'we', 'see', 'to', 'by', 'with', 'has', 'as'}


def _sp(t):
    return t.strip('.,;:()[]"\'')


def _sentences(lo, line):
    out = []
    pos = 0
    for m in SENT_SPLIT.finditer(line):
        out.append((lo + pos, line[pos:m.start()]))
        pos = m.end()
    out.append((lo + pos, line[pos:]))
    return out


def induce_prose(lines, used, out, know, Fact, parse_cell, tmpl_df=0.85, min_sent=3):
    def items_of(slo, st):
        its = []
        for m in TOK.finditer(st):
            raw = m.group(0)
            core = raw.rstrip('.,;')
            inner = core.strip('()[]')
            p = parse_cell(inner, know) if re.search(r'\d', inner) else None
            its.append({'lo': slo + m.start(), 'hi': slo + m.start() + len(core), 'text': core, 'w': _sp(raw).lower(), 'p': p, 'wtxt': _sp(raw), 'raw': raw})
        return its

    blocks, cur = [], []
    for li, (lo, l) in enumerate(lines):
        if used[li] or not l.strip() or l.count('|') >= 2:
            if len(cur) >= min_sent:
                blocks.append(cur)
            cur = []
            continue
        added = False
        for slo, st in _sentences(lo, l):
            its = items_of(slo, st)
            if any(i['p'] is not None for i in its):
                cur.append((slo, st, its))
                added = True
        if not added:
            if len(cur) >= min_sent:
                blocks.append(cur)
            cur = []
    if len(cur) >= min_sent:
        blocks.append(cur)
    for blk in blocks:
        _block(blk, out, know, Fact, parse_cell, tmpl_df)


def _runs(st, slo, its, sepset, numidx):
    """word runs (original substrings) between separator tokens / value tokens"""
    runs, cur = [], []
    for k, i in enumerate(its):
        if k in numidx or i['w'] in sepset or not i['w'] or re.fullmatch(r'[=:;]+', i['w']):
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append(i)
        if i['raw'][-1:] in (',', ';', ':'):
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [st[r[0]['lo'] - slo:r[-1]['lo'] - slo + len(r[-1]['raw'])].rstrip('.,;:') for r in runs]


def _block(blk, out, know, Fact, parse_cell, tmpl_df):
    n = len(blk)
    unit_follow = Counter()
    for slo, st, its in blk:
        for a, b in zip(its, its[1:]):
            if a['p'] is not None and b['p'] is None and re.fullmatch(r'[A-Za-zµμ/%]{1,8}', b['wtxt'] or '-') and b['lo'] == a['hi'] + 1:
                unit_follow[b['w']] += 1
    unit_words = {w for w, c in unit_follow.items() if c >= 3 or w in know.units}
    sent = []
    for slo, st, its in blk:
        merged = []
        k = 0
        while k < len(its):
            it = its[k]
            if it['p'] is not None and k + 1 < len(its) and its[k + 1]['p'] is None and its[k + 1]['w'] in unit_words and its[k + 1]['lo'] == it['hi'] + 1:
                nx = its[k + 1]
                hi = nx['lo'] + len(nx['wtxt'])
                txt = st[it['lo'] - slo:hi - slo]
                p = parse_cell(txt, know)
                if p is not None:
                    merged.append({'lo': it['lo'], 'hi': hi, 'text': txt, 'w': '', 'p': p, 'wtxt': txt, 'raw': st[it['lo'] - slo:nx['lo'] - slo + len(nx['raw'])], 'num': True})
                    k += 2
                    continue
            d = dict(it)
            d['num'] = it['p'] is not None
            merged.append(d)
            k += 1
        sent.append((slo, st, merged))
    df = Counter()
    for _, _, its in sent:
        for w in {i['w'] for i in its if not i['num'] and i['w']}:
            df[w] += 1
    tmpl = {w for w, c in df.items() if c / n >= tmpl_df}

    def sig(its, k):
        pw = its[k - 1]['w'] if k > 0 else '^'
        nw = its[k + 1]['w'] if k + 1 < len(its) else '$'
        return (pw if (pw in tmpl or pw == '^') else '*', nw if (nw in tmpl or nw == '$') else '*')

    sigc = Counter()
    for _, _, its in sent:
        for k, i in enumerate(its):
            if i['num']:
                sigc[sig(its, k)] += 1
    vsig = {sg for sg, c in sigc.items() if c >= 0.6 * n and sg != ('*', '*')}
    if not vsig:
        return
    for _, _, its in sent:
        for k, i in enumerate(its):
            if i['num'] and sig(its, k) not in vsig:
                i['num'] = False
    cnt = Counter(sum(1 for i in its if i['num']) for _, _, its in sent)
    m = cnt.most_common(1)[0][0]
    if m < 1:
        return
    sep_df = tmpl | STOP
    sep_sl = set(STOP)
    cand = {}
    for name, sep in (('df', sep_df), ('sl', sep_sl)):
        rows = []
        for slo, st, its in sent:
            nk = [k for k, i in enumerate(its) if i['num']]
            if len(nk) != m:
                continue
            rows.append(([its[k] for k in nk], _runs(st, slo, its, sep, set(nk))))
        if rows:
            nr = Counter(len(r[1]) for r in rows).most_common(1)[0][0]
            cand[name] = (nr, [r for r in rows if len(r[1]) == nr])
    order = ['df', 'sl'] if m == 1 else ['sl', 'df']
    pick = None
    for name in order:
        if name in cand and cand[name][0] == m + 1 and len(cand[name][1]) >= 3:
            pick = name
            break
    if pick is None:
        return
    nr, rows = cand[pick]
    if m == 1:
        ch = [0, 0]
        for a, b in zip(rows, rows[1:]):
            for p in range(2):
                ch[p] += a[1][p] != b[1][p]
        if ch[0] == ch[1]:
            row_pos, conf = 0, 0.3
        else:
            row_pos, conf = (0, 0.6) if ch[0] < ch[1] else (1, 0.6)
        for nums, runs in rows:
            x = nums[0]
            p = x['p']
            out.append(Fact(rk=norm_key(runs[row_pos]), ck=core_key(runs[1 - row_pos]), v=p['v'], conf=conf * p['conf'], span=(x['lo'], x['hi']), raw=x['text'], hdr=runs[1 - row_pos],
                            notes=p['notes'], struct='induced', base=p['base'], scale=p['scale'], rowtxt=runs[row_pos]))
    else:
        distinct = [len({r[1][p] for r in rows}) for p in range(nr)]
        row_pos = max(range(nr), key=lambda p: distinct[p])
        srt = sorted(distinct)
        conf = 0.55 if srt[-1] > srt[-2] else 0.3
        hdr_pos = [p for p in range(nr) if p != row_pos]
        for nums, runs in rows:
            for x, hp in zip(nums, hdr_pos):
                p = x['p']
                out.append(Fact(rk=norm_key(runs[row_pos]), ck=core_key(runs[hp]), v=p['v'], conf=conf * p['conf'], span=(x['lo'], x['hi']), raw=x['text'], hdr=runs[hp], notes=p['notes'],
                                struct='induced', base=p['base'], scale=p['scale'], rowtxt=runs[row_pos]))
