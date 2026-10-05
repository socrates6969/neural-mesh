"""k3_extract: hand-written text -> (row, column, value) fact extractor for flattened/reflowed tables. Ideas copied from h19 (span offsets, conf from
parse notes, abstain below tau) but re-built for generic tables (h19's scanners were tied to generated fact schemas).
Knowledge levels:  v1 = what the author knew from the DEV split only (units seen there, ASCII minus, US separators)
                   v2 = v1 + fixes derived from DEV-split render failures only (split-continuation, merged header, blank cells, accounting negatives,
                        unicode minus / trailing minus, EU separators, micro-sign variants, unit list) -- frozen before held/confirm."""
import re, csv
from k3_data import strip_md, norm_key, core_key

NUM_US = re.compile(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+')
NUM_EU = re.compile(r'\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+')
UNITISH = re.compile(r'^[A-Za-zµμ/%°²·\-]{1,16}$')


class Know:
    def __init__(self, name, units, paren=False, umin=False, trail=False, eu=False, cur=True, pm=False, layouts=('pipe', 'ws', 'wrap', 'csv'), prose=('F1', 'F2', 'F3'),
                 nolabel_split=False, merged=False, blank_cells=False):
        self.name = name
        self.units = units
        self.paren = paren
        self.umin = umin
        self.trail = trail
        self.eu = eu
        self.cur = cur
        self.pm = pm
        self.layouts = set(layouts)
        self.prose = set(prose)
        self.nolabel_split = nolabel_split
        self.merged = merged
        self.blank_cells = blank_cells
        self.induce = False


V1_UNITS = {'us': 1e-6, 'ms': 1e-3, 's': 1.0, 'b': 1.0, 'kb': 1e3, 'mb': 1e6, 'bytes': 1.0}  # lower-cased keys; '%'/KMB suffix handled separately
V2_UNITS = dict(V1_UNITS)
V2_UNITS.update({'ns': 1e-9, 'µs': 1e-6, 'μs': 1e-6, 'sec': 1.0, 'secs': 1.0, 'gb': 1e9, 'kib': 1024.0, 'mib': 1024.0 ** 2, 'gib': 1024.0 ** 3, 'min': 60.0, 'byte': 1.0})

KNOW_V1 = Know('v1', V1_UNITS)
KNOW_V2 = Know('v2', V2_UNITS, paren=True, umin=True, trail=True, eu=True, pm=True, nolabel_split=True, merged=True, blank_cells=True)


def unit_mult(u, know):
    """multiplier for unit string (numerator before '/'), or None if not known to this extractor"""
    u0 = u.strip().strip('()[]').lower()
    if not u0:
        return 1.0
    a = u0.split('/')[0].strip()
    if a in know.units:
        return know.units[a]
    return None


def header_scale(h, know):
    h0 = h.lower()
    m = re.search(r'\(\s*(?:in\s+)?\$?\s*([a-zµμ%/×]+)\s*\)|\[\s*(?:in\s+)?\$?\s*([a-zµμ%/×]+)\s*\]|,\s*([a-zµμ%/]+)\s*$', h0)
    if m:
        u = next(g for g in m.groups() if g)
        if u == '%':
            return 1e-2
        if u in ('thousands', 'k'):
            return 1e3
        if u in ('millions',) or u == 'mn':
            return 1e6
        mu = unit_mult(u, know)
        if mu is not None and u not in ('x', '×'):
            return mu
    return 1.0


def parse_cell(raw, know, hscale=1.0):
    """-> dict(v, base, scale, conf, notes) or None. v is the value the extractor will report."""
    s = strip_md(raw).replace(' ', ' ').strip()
    if not s:
        return None
    s = re.sub(r'^[~≈≥≤]\s*', '', s)
    neg = False
    notes = []
    if len(s) > 1 and s[0] == '(' and s[-1] == ')':
        if not know.paren:
            return None
        s = s[1:-1].strip()
        neg = True
    if s[:1] == '−':
        if not know.umin:
            return None
        neg = True
        s = s[1:].strip()
    elif s[:1] in ('-', '+'):
        neg = s[0] == '-'
        s = s[1:].strip()
    if know.cur and s[:1] in '$€£':
        s = s[1:].strip()
    if know.trail and re.search(r'\d-$', s):
        neg = True
        s = s[:-1]
    mus = None
    eu = False
    if know.eu:
        m = NUM_EU.match(s)
        if m:
            t = m.group(0)
            # unambiguous EU: comma followed by !=3 digits, or multiple dot groups
            if (',' in t and not re.search(r',\d{3}$', t)) or t.count('.') >= 2:
                mus = t.replace('.', '').replace(',', '.')
                eu = True
                end = m.end()
            elif re.fullmatch(r'\d{1,3}\.\d{3}', t) or re.fullmatch(r'\d+,\d{3}', t):
                mus = None  # ambiguous: fall through to US reading
    if mus is None:
        m = NUM_US.match(s)
        if not m:
            return None
        t = m.group(0)
        mus = t.replace(',', '')
        end = m.end()
    base = float(mus)
    rest = s[end:].strip()
    scale = 1.0
    conf = 1.0
    unit_seen = False
    if rest:
        r = rest
        if know.pm:
            r = re.sub(r'^\s*±\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*', '', r).strip()
        elif r.startswith('±'):
            return None
        if r:
            mm = re.match(r'^(%|[KMBkmb](?![A-Za-z])|[x×](?![A-Za-z]))\s*(.*)$', r)
            if mm:
                suf, r2 = mm.group(1), mm.group(2)
                if suf == '%':
                    scale *= 1e-2
                elif suf in 'x×':
                    pass
                else:
                    scale *= {'k': 1e3, 'm': 1e6, 'b': 1e9}[suf.lower()]
                unit_seen = True
                r = r2.strip()
            if r:
                if not UNITISH.match(r.strip('()')):
                    return None
                mu = unit_mult(r, know)
                if mu is None:
                    conf *= 0.7
                    notes.append('unit_unknown')
                else:
                    scale *= mu
                unit_seen = True
    if not unit_seen and hscale != 1.0:
        scale *= hscale
        notes.append('hdr_scale')
    if eu:
        notes.append('eu')
    v = base * scale * (-1 if neg else 1)
    return {'v': v, 'base': base, 'scale': scale, 'conf': conf, 'notes': notes, 'neg': neg}


# ---------------------------------------------------------------- structure parsers (all return facts with char spans)
def lines_with_offsets(text):
    out = []
    lo = 0
    for l in text.split('\n'):
        out.append((lo, l))
        lo += len(l) + 1
    return out


def split_pipe_offsets(lo, line):
    cells = []
    i = 0
    n = len(line)
    start = 0
    tick = False
    segs = []
    while i <= n:
        if i == n or (line[i] == '|' and not tick and (i == 0 or line[i - 1] != '\\')):
            segs.append((start, i))
            start = i + 1
        elif line[i] == '`':
            tick = not tick
        i += 1
    if segs and line.strip().startswith('|'):
        segs = segs[1:]
    if segs and line.strip().endswith('|'):
        segs = segs[:-1]
    for a, b in segs:
        seg = line[a:b]
        t0 = len(seg) - len(seg.lstrip())
        t1 = len(seg.rstrip())
        if t1 <= t0:
            cells.append((lo + a, lo + a, ''))
        else:
            cells.append((lo + a + t0, lo + a + t1, seg[t0:t1]))
    return cells


SEP = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$')


class Fact:
    __slots__ = ('rk', 'ck', 'v', 'conf', 'span', 'raw', 'hdr', 'notes', 'struct', 'base', 'scale', 'rowtxt')

    def __init__(self, **k):
        for a in self.__slots__:
            setattr(self, a, k.get(a))

    def key(self):
        return (self.rk, self.ck)


STRUCT_CONF = {'pipe': 1.0, 'csv': 0.95, 'ws': 0.9, 'wrap': 0.8, 'F1': 0.9, 'F2': 0.9, 'F3': 0.9, 'split': 0.7, 'merged': 0.4}


def emit_grid(header, rows, struct, know, hdr_conf=1.0, out=None):
    """header: list[str]; rows: list of list[(lo,hi,txt)] (cells incl. label col 0)."""
    out = out if out is not None else []
    ck = [core_key(h) for h in header]
    hs = [header_scale(h, know) for h in header]
    for r in rows:
        if not r:
            continue
        lab = r[0][2]
        rk = norm_key(lab)
        for j in range(1, min(len(r), len(header))):
            lo, hi, txt = r[j]
            p = parse_cell(txt, know, hs[j])
            if p is None:
                continue
            conf = STRUCT_CONF.get(struct, 0.8) * p['conf'] * hdr_conf
            out.append(Fact(rk=rk, ck=ck[j], v=p['v'], conf=conf, span=(lo, hi), raw=txt, hdr=header[j], notes=p['notes'], struct=struct, base=p['base'], scale=p['scale'], rowtxt=lab))
    return out


def csv_offsets(lo, line):
    cells = []
    i = 0
    n = len(line)
    while i <= n:
        if i < n and line[i] == '"':
            j = i + 1
            buf = ''
            while j < n:
                if line[j] == '"':
                    if j + 1 < n and line[j + 1] == '"':
                        buf += '"'
                        j += 2
                        continue
                    break
                buf += line[j]
                j += 1
            cells.append((lo + i + 1, lo + j, buf))
            i = j + 2
        else:
            j = line.find(',', i)
            if j < 0:
                j = n
            cells.append((lo + i, lo + j, line[i:j].strip()))
            i = j + 1
        if i > n:
            break
    return cells


WS_FIELD = re.compile(r'\S+(?: \S+)*')


def ws_fields(lo, line):
    return [(lo + m.start(), lo + m.end(), m.group(0)) for m in WS_FIELD.finditer(line)]


def is_numeric_field(txt, know):
    return parse_cell(txt, know) is not None


def extract(text, know, layout_hint=None):
    """returns list[Fact]. Structures are consumed in order pipe -> prose families -> csv -> ws (wrap/split/blank handling by know flags)."""
    lines = lines_with_offsets(text)
    used = [False] * len(lines)
    out = []
    # ---- pipe tables
    i = 0
    while i < len(lines) - 1:
        lo, l = lines[i]
        if l.count('|') >= 2 and i + 1 < len(lines) and SEP.match(lines[i + 1][1]) and '-' in lines[i + 1][1]:
            hdr = [c[2] for c in split_pipe_offsets(lo, l)]
            j = i + 2
            rows = []
            while j < len(lines) and lines[j][1].count('|') >= 2:
                rows.append(split_pipe_offsets(lines[j][0], lines[j][1]))
                used[j] = True
                j += 1
            used[i] = used[i + 1] = True
            emit_grid(hdr, rows, 'pipe', know, out=out)
            i = j
        else:
            i += 1
    # ---- prose families known to this extractor
    fam = know.prose
    for li, (lo, l) in enumerate(lines):
        if used[li]:
            continue
        if 'F3' in fam:
            m = re.match(r'^- (?P<row>.+?) / (?P<col>.+?): (?P<val>.+)$', l)
            if m:
                _emit_prose(out, m, lo, 'F3', know, l)
                used[li] = True
                continue
        if 'F1' in fam and ' = ' in l:
            m = re.match(r'^(?P<row>[^:]{1,100}): (?P<body>.+)\.$', l)
            if m:
                row = m.group('row')
                bstart = m.start('body')
                pos = 0
                body = m.group('body')
                for part in re.split(r'; ', body):
                    k = body.find(part, pos)
                    pos = k + len(part)
                    mm = re.match(r'^(?P<col>.+?) = (?P<val>.+)$', part)
                    if not mm:
                        continue
                    vlo = lo + bstart + k + mm.start('val')
                    _add_prose(out, row, mm.group('col'), mm.group('val'), (vlo, vlo + len(mm.group('val'))), 'F1', know)
                used[li] = True
                continue
        if 'F2' in fam and l.startswith('For ') and ', the ' in l:
            for m in re.finditer(r'For (?P<row>.+?), the (?P<col>.+?) is (?P<val>.+?)\.(?=\s+For\s|\s*$)', l):
                vlo = lo + m.start('val')
                _add_prose(out, m.group('row'), m.group('col'), m.group('val'), (vlo, vlo + len(m.group('val'))), 'F2', know)
            used[li] = True
    # ---- csv blocks (>=3 fields, >=3 consecutive lines with equal field count)
    li = 0
    while li < len(lines):
        if used[li] or lines[li][1].count(',') < 2:
            li += 1
            continue
        cnt = len(csv_offsets(lines[li][0], lines[li][1]))
        j = li
        grp = []
        while j < len(lines) and not used[j] and lines[j][1].count(',') >= 2 and len(csv_offsets(lines[j][0], lines[j][1])) == cnt:
            grp.append(csv_offsets(lines[j][0], lines[j][1]))
            j += 1
        if len(grp) >= 3 and cnt >= 3 and any(is_numeric_field(c[2], know) for r in grp[1:] for c in r[1:]):
            hdr = [c[2] for c in grp[0]]
            emit_grid(hdr, grp[1:], 'csv', know, out=out)
            for k in range(li, j):
                used[k] = True
            li = j
        else:
            li += 1
    # ---- whitespace-aligned blocks (incl. wrapped headers, continuation blocks)
    _ws_blocks(lines, used, out, know)
    if know.merged:
        _merged_blocks(lines, used, out, know)
    if know.induce:
        from k3_induce import induce_prose
        induce_prose(lines, used, out, know, Fact, parse_cell)
    return out


def _add_prose(out, row, col, val, span, struct, know):
    p = parse_cell(val.strip(), know, header_scale(col, know))
    if p is None:
        return
    out.append(Fact(rk=norm_key(row), ck=core_key(col), v=p['v'], conf=STRUCT_CONF[struct] * p['conf'], span=span, raw=val.strip(), hdr=col, notes=p['notes'], struct=struct, base=p['base'], scale=p['scale'], rowtxt=row))


def _emit_prose(out, m, lo, struct, know, line):
    _add_prose(out, m.group('row'), m.group('col'), m.group('val'), (lo + m.start('val'), lo + m.end('val')), struct, know)


def _ws_blocks(lines, used, out, know):
    from collections import Counter
    n = len(lines)
    fields = [ws_fields(lo, l) if (not used[i] and l.strip()) else [] for i, (lo, l) in enumerate(lines)]
    i = 0
    prev_grid = None  # for nolabel continuation
    while i < n:
        if used[i] or len(fields[i]) < 2:
            i += 1
            continue
        j = i
        while j < n and not used[j] and (len(fields[j]) >= 2 or (len(fields[j]) == 1 and len(fields[j][0][2]) <= 14)):
            j += 1
        run = list(range(i, j))
        numrows = [k for k in run[1:] if any(is_numeric_field(f[2], know) for f in fields[k][1:])]
        if len(numrows) < 2:
            i = j
            continue
        ncol = Counter(len(fields[k]) for k in numrows).most_common(1)[0][0]
        # header zone: first line of the run + continuation fragment lines
        hdr_idx = [run[0]]
        k = 1
        while k < len(run) and len(hdr_idx) < 8:
            fk = fields[run[k]]
            numeric = any(is_numeric_field(f[2], know) for f in fk)
            short = all(len(f[2]) <= 12 for f in fk)
            if (not numeric) and (len(fk) < ncol or short) and len(fk) <= ncol and len(hdr_idx) < len(run) - 2:
                hdr_idx.append(run[k])
                k += 1
            else:
                break
        first = run[k] if k < len(run) else None
        if first is None:
            i = j
            continue
        data = [r for r in run[k:] if len(fields[r]) == ncol]
        if len(data) < 2:
            i = j
            continue
        ext = []
        for c in range(ncol):
            ext.append((min(fields[r][c][0] - lines[r][0] for r in data), max(fields[r][c][1] - lines[r][0] for r in data)))
        hdr_frag = [[] for _ in range(ncol)]
        for hk in hdr_idx:
            for f in fields[hk]:
                best, bo = None, 0
                f0, f1 = f[0] - lines[hk][0], f[1] - lines[hk][0]
                for c in range(ncol):
                    ov = min(f1, ext[c][1]) - max(f0, ext[c][0])
                    if ov > bo:
                        best, bo = c, ov
                if best is None:
                    ctr = (f0 + f1) / 2
                    best = min(range(ncol), key=lambda c: abs((ext[c][0] + ext[c][1]) / 2 - ctr))
                hdr_frag[best].append(f[2])
        header = [' '.join(fr) for fr in hdr_frag]
        wrapped = len(hdr_idx) > 1
        rows = []
        consumed = list(hdr_idx)
        for r in run[k:]:
            if len(fields[r]) == ncol:
                rows.append(fields[r])
                consumed.append(r)
            elif know.blank_cells and len(fields[r]) >= 2:
                row = [(0, 0, '')] * ncol
                ok = True
                for f in fields[r]:
                    best, bo = None, 0
                    f0, f1 = f[0] - lines[r][0], f[1] - lines[r][0]
                    for c in range(ncol):
                        ov = min(f1, ext[c][1]) - max(f0, ext[c][0])
                        if ov > bo:
                            best, bo = c, ov
                    if best is None or row[best][2]:
                        ok = False
                        break
                    row[best] = f
                if ok:
                    rows.append(row)
                    consumed.append(r)
        rows = [r for r in rows if r[0][2]]
        if rows and all(is_numeric_field(r[0][2], know) for r in rows) and know.nolabel_split and prev_grid is not None and len(rows) == len(prev_grid[1]):
            pg_header, pg_rows = prev_grid
            rows2 = [[pr[0]] + list(nr) for pr, nr in zip(pg_rows, rows)]
            emit_grid(['row'] + header, rows2, 'split', know, hdr_conf=0.9, out=out)
        else:
            emit_grid(header, rows, 'wrap' if wrapped else 'ws', know, hdr_conf=0.9 if wrapped else 1.0, out=out)
            prev_grid = (header, rows)
        for r in consumed:
            used[r] = True
        i = j


def _merged_blocks(lines, used, out, know):
    """single-space separated rows: label tokens + k trailing numeric tokens; header line above split into k+1 groups (guess)."""
    n = len(lines)
    i = 0
    while i < n:
        if used[i] or not lines[i][1].strip():
            i += 1
            continue
        def row_split(l):
            toks = l.split()
            k = 0
            while k < len(toks) and parse_cell(toks[len(toks) - 1 - k], know) is not None:
                k += 1
            return toks, k
        toks, k = row_split(lines[i][1])
        if k < 1 or len(toks) <= k:
            i += 1
            continue
        j = i
        rows = []
        while j < n and not used[j] and lines[j][1].strip():
            t2, k2 = row_split(lines[j][1])
            if k2 != k or len(t2) <= k2:
                break
            rows.append((j, t2))
            j += 1
        if len(rows) >= 2 and i >= 1 and not used[i - 1] and lines[i - 1][1].strip():
            hline = lines[i - 1][1]
            if row_split(hline)[1] == 0:
                words = hline.split()
                # attach '(unit)' tokens to previous word
                groups = []
                for w in words:
                    if w.startswith('(') and groups:
                        groups[-1] += ' ' + w
                    else:
                        groups.append(w)
                hdr = None
                conf = 0.4
                if len(groups) == k + 1:
                    hdr = groups
                elif len(groups) > k + 1:
                    # capitalisation breaks
                    caps = []
                    for w in groups:
                        if w[:1].isupper() or not caps:
                            caps.append(w)
                        else:
                            caps[-1] += ' ' + w
                    if len(caps) == k + 1:
                        hdr = caps
                        conf = 0.5
                if hdr is not None:
                    erows = []
                    pos_cache = {}
                    for (jj, t2) in rows:
                        lo, l = lines[jj]
                        # char offsets of tokens
                        offs = [(m.start() + lo, m.end() + lo, m.group(0)) for m in re.finditer(r'\S+', l)]
                        label = ' '.join(t[2] for t in offs[:len(offs) - k])
                        cells = [(offs[0][0], offs[len(offs) - k - 1][1], label)] + offs[len(offs) - k:]
                        erows.append(cells)
                    emit_grid(hdr, erows, 'merged', know, hdr_conf=conf / STRUCT_CONF['merged'] if conf > STRUCT_CONF['merged'] else 1.0, out=out)
                    for (jj, _) in rows:
                        used[jj] = True
                    used[i - 1] = True
        i = max(j, i + 1)


import copy
KNOW_V3 = copy.copy(KNOW_V2)
KNOW_V3.name = 'v3'
KNOW_V3.induce = True
KNOW_IND = copy.copy(KNOW_V2)
KNOW_IND.name = 'ind'
KNOW_IND.induce = True
KNOW_IND.prose = set()
