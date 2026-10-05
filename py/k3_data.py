"""k3_data: collect REAL markdown tables from disk (read-only), parse the cell grid (pipe syntax) and derive GOLD numeric values.
Gold = the markdown pipe grid; gold value parser is a separate, fuller implementation than the extractor's (extractor only knows the DEV unit list).
No network. Sources: neural-mesh/results/*.md, DESIGN.md (repo) and ~/.cargo/registry/src/*/<crate>/**.md (vendored crates)."""
import glob, os, re, sys, json, collections

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
REG = glob.glob(os.path.expanduser('~/.cargo/registry/src/*'))


def fnv(s):
    h = 0xcbf29ce484222325
    for b in s.encode('utf8', 'ignore'):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def strip_md(s):
    s = s.strip()
    s = re.sub(r'(\*\*|__|`|\*)', '', s)
    s = re.sub(r'<[^>]+>', '', s)
    return s.strip()


def norm_key(s):
    s = strip_md(s).lower()
    s = re.sub(r'[^0-9a-z%µμ/ ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def core_key(h):
    """column key: header text with parenthesised / bracketed unit annotations and trailing ', unit' removed"""
    t = strip_md(h)
    c = re.sub(r'\([^)]*\)|\[[^\]]*\]', ' ', t)
    c = re.sub(r',\s*[A-Za-zµμ%/]{1,6}\s*$', ' ', c)
    k = norm_key(c)
    return k if k else norm_key(t)


# ---------------------------------------------------------------- gold value parsing (FULL unit table)
UNIT_MULT = {  # -> base unit (seconds, bytes, plain)
    'ns': 1e-9, 'us': 1e-6, 'µs': 1e-6, 'μs': 1e-6, 'ms': 1e-3, 's': 1.0, 'sec': 1.0, 'secs': 1.0, 'min': 60.0, 'h': 3600.0,
    'b': 1.0, 'kb': 1e3, 'mb': 1e6, 'gb': 1e9, 'tb': 1e12, 'kib': 1024.0, 'mib': 1024.0 ** 2, 'gib': 1024.0 ** 3, 'tib': 1024.0 ** 4,
    '%': 1e-2, 'x': 1.0, '×': 1.0, 'ops': 1.0, 'op': 1.0, 'qps': 1.0, 'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9,
    'bytes': 1.0, 'byte': 1.0, 'mw': 1e-3, 'w': 1.0, 'uj': 1e-6, 'µj': 1e-6, 'nj': 1e-9, 'mj': 1e-3, 'j': 1.0, 'pj': 1e-12, 'fj': 1e-15,
    'mv': 1e-3, 'v': 1.0, 'uv': 1e-6, 'µv': 1e-6, 'ma': 1e-3, 'a': 1.0, 'ua': 1e-6, 'µa': 1e-6, 'pa': 1e-12, 'na': 1e-9,
    'mm': 1e-3, 'um': 1e-6, 'µm': 1e-6, 'nm': 1e-9, 'cm': 1e-2, 'km': 1e3,
    'gflops': 1e9, 'mflops': 1e6, 'tflops': 1e12, 'kflops': 1e3, 'flops': 1.0, 'tokens': 1.0, 'items': 1.0, 'rows': 1.0, 'cells': 1.0,
    'mpix': 1e6, 'kloc': 1e3, 'loc': 1.0,
    'nsec': 1e-9, 'usec': 1e-6, 'msec': 1e-3, 'seconds': 1.0, 'second': 1.0, 'minutes': 60.0, 'minute': 60.0, 'hours': 3600.0, 'hour': 3600.0, 'days': 86400.0,
}
SUFFIX = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}
NUM = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
NUM_RE = re.compile(r'^(?P<neg>[-−–+]?)\s*(?P<cur>[$€£¥]?)\s*(?P<num>' + NUM + r')(?P<rest>.*)$')


def gold_unit_mult(u):
    u0 = u.strip().strip('()[]').lower().replace('per ', '/')
    if not u0:
        return 1.0, True
    a = u0.split('/')[0].strip()
    if a in UNIT_MULT:
        return UNIT_MULT[a], True
    return 1.0, False


def header_unit(h):
    h0 = h.lower()
    m = re.search(r'\(\s*(?:in\s+)?\$?\s*([a-zµμ%/×]+)\s*\)|\[\s*(?:in\s+)?\$?\s*([a-zµμ%/×]+)\s*\]|,\s*([a-zµμ%/]+)\s*$', h0)
    if m:
        u = next(g for g in m.groups() if g)
        if u in ('thousands', 'k'):
            return 1e3, 'k'
        if u in ('millions', 'mm', 'mn', 'm') and u != 'ms':
            return 1e6, 'm'
        if u in ('billions', 'bn'):
            return 1e9, 'b'
        mult, known = gold_unit_mult(u)
        if known and u not in ('x', '×', 'ops', 'op', 'items', 'rows', 'tokens', 'cells'):
            return mult, u
    if re.search(r"in thousands|\$000|\(000s\)|'000", h0):
        return 1e3, 'k'
    if re.search(r'in millions', h0):
        return 1e6, 'm'
    return 1.0, None


def gold_parse_cell(cell, header=''):
    """gold semantic parse. returns dict(value, unit) or None if not a plain numeric cell."""
    s = strip_md(cell).replace(' ', ' ').strip()
    if not s:
        return None
    s = re.sub(r'^[~≈≥≤]\s*', '', s)
    paren = False
    if re.fullmatch(r'\(.*\)', s):
        paren = True
        s = s[1:-1].strip()
    neg_trail = False
    if s.endswith('-') and len(s) > 1 and s[-2].isdigit():
        neg_trail = True
        s = s[:-1]
    m = NUM_RE.match(s)
    if not m:
        return None
    num = float(m.group('num').replace(',', ''))
    rest = m.group('rest').strip()
    neg = m.group('neg') in ('-', '−', '–') or paren or neg_trail
    mult = 1.0
    usrc = None
    if rest:
        r = re.sub(r'^\s*±\s*(' + NUM + r').*$', '', rest).strip()
        if r:
            mm = re.match(r'^(%|[KMBTkmbt](?![a-zA-Z])|[x×](?![a-zA-Z]))\s*(.*)$', r)
            if mm:
                suf, r2 = mm.group(1), mm.group(2)
                if suf == '%':
                    mult *= 1e-2
                    usrc = '%'
                elif suf in 'x×':
                    pass
                else:
                    mult *= SUFFIX[suf.lower()]
                    usrc = suf
                r = r2.strip()
            if r:
                if not re.fullmatch(r'\(?[A-Za-zµμ/%°²·\-]{1,16}\)?', r):
                    return None
                mu, known = gold_unit_mult(r)
                if known:
                    mult *= mu
                    usrc = (usrc or '') + r
                elif re.fullmatch(r'[A-Za-zµμ/%°²·\-]{1,14}', r):
                    usrc = (usrc or '') + r  # unknown unit word: value as written
                else:
                    return None
    if usrc is None:
        hm, hl = header_unit(header)
        if hl:
            mult *= hm
            usrc = 'hdr:' + hl
    v = num * mult * (-1 if neg else 1)
    return {'value': v, 'unit': usrc, 'raw': cell.strip(), 'aw': num * (-1 if neg else 1)}


# ---------------------------------------------------------------- markdown table extraction
SEP_RE = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?\s*$')


def split_row(line):
    l = line.strip()
    if l.startswith('|'):
        l = l[1:]
    if l.endswith('|'):
        l = l[:-1]
    cells, cur, tick = [], '', False
    i = 0
    while i < len(l):
        c = l[i]
        if c == '`':
            tick = not tick
        if c == '\\' and i + 1 < len(l) and l[i + 1] == '|':
            cur += '|'
            i += 2
            continue
        if c == '|' and not tick:
            cells.append(cur)
            cur = ''
        else:
            cur += c
        i += 1
    cells.append(cur)
    return [c.strip() for c in cells]


def find_tables(text):
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines) - 1:
        if '|' in lines[i] and SEP_RE.match(lines[i + 1]) and '-' in lines[i + 1]:
            hdr = split_row(lines[i])
            j = i + 2
            rows = []
            while j < len(lines) and '|' in lines[j] and lines[j].strip():
                rows.append(split_row(lines[j]))
                j += 1
            if len(hdr) >= 2 and len(rows) >= 2:
                out.append({'start': i, 'end': j, 'header': hdr, 'rows': rows})
            i = j
        else:
            i += 1
    return out, lines


def build_table(tb, lines, doc_id):
    hdr = tb['header']
    ncol = len(hdr)
    rows = [(r + [''] * ncol)[:ncol] for r in tb['rows']]
    hk = [core_key(h) for h in hdr]
    rk = [norm_key(r[0]) for r in rows]
    why = None
    if len(set(rk)) != len(rk) or '' in rk:
        why = 'dup_or_empty_row_key'
    elif len(set(hk[1:])) != len(hk[1:]) or '' in hk[1:]:
        why = 'dup_or_empty_col_key'
    facts = {}
    for ri, r in enumerate(rows):
        for ci in range(1, ncol):
            g = gold_parse_cell(r[ci], hdr[ci])
            if g is not None:
                facts[(rk[ri], hk[ci])] = {'value': g['value'], 'unit': g['unit'], 'raw': r[ci], 'ri': ri, 'ci': ci, 'aw': g['aw']}
    ctx_before = lines[max(0, tb['start'] - 3):tb['start']]
    ctx_after = lines[tb['end']:tb['end'] + 3]
    return {'doc': doc_id, 'header': hdr, 'rows': rows, 'rk': rk, 'hk': hk, 'facts': facts, 'why_skip': why,
            'ctx_before': ctx_before, 'ctx_after': ctx_after, 'nnum': len(facts), 'lineno': tb['start']}


def split_of(path, source):
    if source == 'registry':
        return 'confirm'
    return 'dev' if fnv(os.path.basename(path)) % 2 == 0 else 'held'


def load_all():
    srcs = []
    for p in sorted(glob.glob(os.path.join(REPO, 'results', '*.md'))) + [os.path.join(REPO, 'DESIGN.md')]:
        if os.path.exists(p):
            srcs.append((p, 'repo'))
    for reg in REG:
        for crate in sorted(os.listdir(reg)):
            d = os.path.join(reg, crate)
            for p in sorted(glob.glob(d + '/*.md') + glob.glob(d + '/docs/*.md') + glob.glob(d + '/*/*.md')):
                srcs.append((p, 'registry'))
    tables = []
    stats = {'files': 0, 'raw_tables': 0}
    seen = set()
    for p, srcname in srcs:
        if p in seen:
            continue
        seen.add(p)
        try:
            t = open(p, encoding='utf8', errors='ignore').read()
        except Exception:
            continue
        tbs, lines = find_tables(t)
        if not tbs:
            continue
        stats['files'] += 1
        stats['raw_tables'] += len(tbs)
        if srcname == 'repo':
            rel = p.replace(REPO, '').replace('\\', '/')
            crate = 'neural-mesh'
        else:
            rel = p.split('index.crates.io-')[-1].split('\\', 1)[-1].replace('\\', '/')
            crate = rel.split('/')[0]
        for ti, tb in enumerate(tbs):
            T = build_table(tb, lines, rel)
            T['source'] = srcname
            T['crate'] = crate
            T['split'] = split_of(p, srcname)
            T['id'] = f'{rel}#{ti}'
            tables.append(T)
    return tables, stats


def usable(T, min_num=2):
    return T['why_skip'] is None and T['nnum'] >= min_num


def corpus_hash(tables):
    h = 0
    for T in tables:
        if usable(T):
            h ^= fnv(T['id'] + str(sorted((k, round(v['value'], 12)) for k, v in T['facts'].items())))
    return '%016x' % h


if __name__ == '__main__':
    tables, st = load_all()
    print('files with tables', st['files'], 'raw tables', st['raw_tables'])
    c = collections.Counter(); n = collections.Counter(); sk = collections.Counter()
    docs = collections.defaultdict(set)
    for T in tables:
        if usable(T):
            c[T['split']] += 1
            n[T['split']] += T['nnum']
            docs[T['split']].add(T['doc'])
        else:
            sk[(T['why_skip'] or 'lt2_numeric_cells')] += 1
    print('usable tables per split', dict(c))
    print('gold numeric cells per split', dict(n))
    print('docs per split', {k: len(v) for k, v in docs.items()})
    print('skipped', dict(sk))
    print('corpus fnv', corpus_hash(tables))
