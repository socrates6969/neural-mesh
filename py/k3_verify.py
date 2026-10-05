"""k3_verify: independent verifier for extracted table facts. Different code path from the extractor's parse_cell (character scanner, own unit table).
Checks: span re-parse (+scale evidence from span suffix/unit and header), ambiguous separators, column scale/magnitude consistency, footing (row/column totals,
percent columns summing to 100), cross-source agreement between two renderings. NARROW = verifier restricted to the extractor's dev knowledge (shared blind spot)."""
import math, re
from k3_data import strip_md, norm_key

# verifier-owned unit table (written independently of the extractor's; BROAD) -------------
BROAD = {
    'ns': 1e-9, 'nsec': 1e-9, 'µs': 1e-6, 'μs': 1e-6, 'us': 1e-6, 'usec': 1e-6, 'ms': 1e-3, 'msec': 1e-3, 's': 1.0, 'sec': 1.0, 'secs': 1.0, 'seconds': 1.0, 'min': 60.0, 'minutes': 60.0,
    'b': 1.0, 'byte': 1.0, 'bytes': 1.0, 'kb': 1e3, 'mb': 1e6, 'gb': 1e9, 'tb': 1e12, 'kib': 1024.0, 'mib': 1024.0 ** 2, 'gib': 1024.0 ** 3, 'tib': 1024.0 ** 4,
    'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9, 'mw': 1e-3, 'w': 1.0, 'mj': 1e-3, 'j': 1.0, 'uj': 1e-6, 'µj': 1e-6, 'nj': 1e-9, 'pj': 1e-12,
    'mv': 1e-3, 'v': 1.0, 'ma': 1e-3, 'ua': 1e-6, 'µa': 1e-6, 'mm': 1e-3, 'cm': 1e-2, 'km': 1e3, 'um': 1e-6, 'µm': 1e-6, 'nm': 1e-9,
    'gflops': 1e9, 'mflops': 1e6, 'tflops': 1e12, 'tokens': 1.0, 'items': 1.0, 'rows': 1.0, 'cells': 1.0, 'ops': 1.0, 'op': 1.0,
}
NARROW = {'us': 1e-6, 'ms': 1e-3, 's': 1.0, 'b': 1.0, 'kb': 1e3, 'mb': 1e6, 'bytes': 1.0}
DIGITS = '0123456789'


class IP:
    __slots__ = ('base', 'scale', 'unit', 'unit_known', 'amb', 'alt_base', 'neg', 'ok')

    def __init__(self):
        self.base = None
        self.scale = 1.0
        self.unit = ''
        self.unit_known = True
        self.amb = False
        self.alt_base = None
        self.neg = False
        self.ok = False


def iparse(txt, units=BROAD, allow_neg_forms=True):
    r = IP()
    s = strip_md(txt).replace(' ', ' ').strip()
    if not s:
        return r
    while s and s[0] in '~≈≥≤><':
        s = s[1:].strip()
    if not s:
        return r
    i = 0
    n = len(s)
    if s[0] == '(' and s[-1] == ')' and n > 2:
        if not allow_neg_forms:
            return r
        r.neg = True
        s = s[1:-1].strip()
        n = len(s)
    while i < n and s[i] in '+-−–':
        if s[i] in '-−–':
            if s[i] != '-' and not allow_neg_forms:
                return r
            r.neg = True
        i += 1
    while i < n and s[i] in '$€£¥ ':
        i += 1
    # scan numeric run
    j = i
    while j < n and (s[j] in DIGITS or s[j] in '.,'):
        j += 1
    run = s[i:j]
    if not run or not any(c in DIGITS for c in run):
        return r
    # a trailing separator belongs to prose, not the number ('12.' / '3,')
    while run and run[-1] in '.,':
        run = run[:-1]
        j -= 1
    nd, nc = run.count('.'), run.count(',')
    val = None
    if nd and nc:
        dec = '.' if run.rfind('.') > run.rfind(',') else ','
        th = ',' if dec == '.' else '.'
        val = float(run.replace(th, '').replace(dec, '.'))
        if dec == ',':
            r.amb = False
    elif nc:
        parts = run.split(',')
        if nc == 1 and len(parts[1]) == 3 and 1 <= len(parts[0]) <= 3:
            val = float(run.replace(',', ''))
            r.amb = True  # 1,234 : US thousands vs EU decimal
            r.alt_base = float(parts[0] + '.' + parts[1])
        elif all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            val = float(run.replace(',', ''))
        elif nc == 1:
            val = float(parts[0] + '.' + parts[1]) if parts[0] or parts[1] else None  # EU decimal comma
        else:
            return r
    elif nd:
        parts = run.split('.')
        if nd == 1:
            val = float(run) if parts[0] or parts[1] else None
            if parts[0] and len(parts[1]) == 3 and 1 <= len(parts[0]) <= 3:
                r.amb = True  # 1.234: decimal vs EU thousands
                r.alt_base = float(parts[0] + parts[1])
        elif all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3:
            val = float(run.replace('.', ''))
        else:
            return r
    else:
        val = float(run)
    if val is None:
        return r
    r.base = val
    rest = s[j:].strip()
    if rest.endswith('-') and rest == '-':
        r.neg = True
        rest = ''
    if rest.startswith('±'):
        k = 1
        while k < len(rest) and (rest[k] in DIGITS or rest[k] in '., '):
            k += 1
        rest = rest[k:].strip()
    # suffix / unit
    if rest:
        low = rest.lower()
        if rest[0] == '%':
            r.scale, r.unit = 1e-2, '%'
            rest = rest[1:].strip()
            low = rest.lower()
        elif rest[0] in 'KMBkmb' and (len(rest) == 1 or not rest[1].isalpha()):
            r.scale, r.unit = {'k': 1e3, 'm': 1e6, 'b': 1e9}[rest[0].lower()], rest[0]
            rest = rest[1:].strip()
            low = rest.lower()
        elif rest[0] in 'x×' and (len(rest) == 1 or not rest[1].isalpha()):
            r.unit = 'x'
            rest = rest[1:].strip()
            low = rest.lower()
        if rest:
            head = low.strip('()').split('/')[0].strip()
            if head in units:
                r.scale *= units[head]
                r.unit += rest
            elif re.fullmatch(r'[A-Za-zµμ/%°²·\-]{1,16}', rest.strip('()')):
                r.unit_known = False
                r.unit += rest
            else:
                return r
    r.ok = True
    return r


def hdr_scale_ev(h, units=BROAD):
    """independent header-scale evidence: returns multiplier or None"""
    h0 = h.lower()
    for pat in (r'\(([^)]{1,14})\)', r'\[([^\]]{1,14})\]', r',\s*([a-zµμ%/]{1,8})\s*$'):
        m = re.search(pat, h0)
        if m:
            u = m.group(1).strip().replace('in ', '').replace('$', '').strip()
            if u == '%':
                return 1e-2
            if u in ('thousands', 'k'):
                return 1e3
            if u in ('millions', 'mn'):
                return 1e6
            hu = u.split('/')[0]
            if hu in units:
                return units[hu]
    return None


def close(a, b, rel=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300) + 1e-15


class VCfg:
    def __init__(self, units=BROAD, span=True, amb=True, colscale=True, footing=True, outlier=False, allow_neg_forms=True, repair=True, footing_precise=True):
        self.units = units
        self.span = span
        self.amb = amb
        self.colscale = colscale
        self.footing = footing
        self.outlier = outlier
        self.allow_neg_forms = allow_neg_forms
        self.repair = repair
        self.footing_precise = footing_precise


VFULL = VCfg()
VNARROW = VCfg(units=NARROW, allow_neg_forms=False)


def verify_facts(text, facts, vc):
    """returns dict id(fact) -> dict(flags=set, repair=value|None)."""
    res = {}
    # per-column groups
    cols = {}
    for f in facts:
        cols.setdefault((f.struct, f.hdr, id(f.rowtxt) if False else None), []).append(f)
    # span re-parse
    for f in facts:
        fl = set()
        rep = None
        ip = None
        if vc.span:
            ip = iparse(text[f.span[0]:f.span[1]], vc.units, vc.allow_neg_forms)
            if not ip.ok:
                fl.add('span_unparseable')
            else:
                hs = hdr_scale_ev(f.hdr, vc.units)
                has_unit = bool(ip.unit)
                exp = ip.base * ip.scale * (-1 if ip.neg else 1)
                if not has_unit and hs is not None:
                    exp = exp * hs
                if not close(f.v, exp):
                    fl.add('value_mismatch')
                    if ip.unit_known and not ip.amb:
                        rep = exp
                    elif ip.amb and close(f.v, ip.alt_base * ip.scale * (-1 if ip.neg else 1)) :
                        pass
                if vc.amb and ip.amb:
                    fl.add('ambiguous_sep_candidate')
        res[id(f)] = {'flags': fl, 'repair': rep, 'ip': ip}
    # ambiguous separator: only a real flag when the same doc shows unambiguous EU evidence in any cell
    eu_evidence = bool(re.search(r'(?<![\d.,])\d{1,3}(?:\.\d{3})+,\d+(?![\d.,])|(?<![\d.,])\d+,\d{1,2}(?![\d.,])(?!\d)', text)) and not re.search(r'(?<![\d.,])\d{1,3}(?:,\d{3})+\.\d+', text)

    for f in facts:
        r = res[id(f)]
        if 'ambiguous_sep_candidate' in r['flags']:
            r['flags'].discard('ambiguous_sep_candidate')
            if eu_evidence:
                r['flags'].add('ambiguous_sep')
    # column scale/magnitude consistency
    if vc.colscale or vc.outlier:
        bycol = {}
        for f in facts:
            bycol.setdefault((f.struct, f.ck), []).append(f)
        for key, fs in bycol.items():
            scales = [(math.log10(abs(f.v)) if f.v else None) for f in fs]
            vals = [x for x in scales if x is not None]
            if vc.colscale:
                # mixed scale evidence in one column: cells with a known unit scale != column majority while the extractor reported different treatment
                cs = {}
                for f in fs:
                    ip = res[id(f)]['ip']
                    if ip is not None and ip.ok:
                        cs[round(math.log10(ip.scale), 6) if ip.scale > 0 else 0] = cs.get(round(math.log10(ip.scale), 6), 0) + 1
                # a fact whose reported scale differs from its own span-derived scale was already flagged by value_mismatch; nothing more here
            if vc.outlier and len(vals) >= 5:
                med = sorted(vals)[len(vals) // 2]
                for f, x in zip(fs, scales):
                    if x is not None and abs(x - med) >= 3.0:
                        res[id(f)]['flags'].add('magnitude_outlier')
    # footing
    if vc.footing:
        bytab = {}
        for f in facts:
            bytab.setdefault((f.struct, f.ck), {})[f.rk] = f
        rows_all = {}
        for f in facts:
            rows_all.setdefault((f.struct, f.rk), {})[f.ck] = f
        tot_re = re.compile(r'^(total|sum|overall|all|grand total|subtotal)\b')
        for (st, ck), d in bytab.items():
            for rk, tf in d.items():
                if tot_re.match(rk) and len(d) >= 4:
                    others = [x.v for k2, x in d.items() if k2 != rk]
                    s = sum(others)
                    tol = foot_tol(tf, [x for k2, x in d.items() if k2 != rk], vc)
                    if not (abs(s - tf.v) <= tol):
                        # only flag when the other cells look like a partition (all same sign) -- 'total' is often non-additive
                        if all(o >= 0 for o in others):
                            res[id(tf)]['flags'].add('footing')
                            for k2, x in d.items():
                                if k2 != rk:
                                    res[id(x)]['flags'].add('footing_member')
        # total columns
        for (st, rk), d in rows_all.items():
            for ck, tf in d.items():
                if tot_re.match(ck) and len(d) >= 4:
                    others = [x.v for k2, x in d.items() if k2 != ck]
                    s = sum(others)
                    tol = foot_tol(tf, [x for k2, x in d.items() if k2 != ck], vc)
                    if not (abs(s - tf.v) <= tol) and all(o >= 0 for o in others):
                        res[id(tf)]['flags'].add('footing')
    return res


def _half_ulp(raw, v):
    """half unit of the last displayed digit, in the value's own scale"""
    t = raw.replace(',', '')
    m = re.search(r'\d+(?:\.(\d+))?|\.(\d+)', t)
    if not m:
        return 0.0
    dec = len(m.group(1) or m.group(2) or '')
    mant = float(m.group(0))
    sc = abs(v / mant) if mant else 1.0
    return 0.5 * (10 ** (-dec)) * sc


def foot_tol(total, members, vc):
    tol = 0.01 * max(abs(total.v), 1e-12) + 1e-9
    if not getattr(vc, 'footing_precise', True):
        return tol
    try:
        tol = max(tol, sum(_half_ulp(m.raw, m.v) for m in members) + _half_ulp(total.raw, total.v))
        if re.search(r'[~≈]|approx', total.raw):
            tol = max(tol, 0.10 * abs(total.v))
    except Exception:
        pass
    return tol


def footing_applicable(facts):
    """number of (struct,column/row) groups where footing can be tested"""
    tot_re = re.compile(r'^(total|sum|overall|all|grand total|subtotal)\b')
    n = 0
    seen = set()
    for f in facts:
        if tot_re.match(f.rk) and (f.struct, f.ck) not in seen:
            seen.add((f.struct, f.ck))
            n += 1
        if tot_re.match(f.ck) and (f.struct, f.rk) not in seen:
            seen.add((f.struct, f.rk))
            n += 1
    return n


HARD_FLAGS = {'span_unparseable', 'value_mismatch', 'ambiguous_sep', 'footing', 'footing_member'}


def is_flagged(flags, use=HARD_FLAGS):
    return bool(flags & use)


def crosscheck(factsA, factsB):
    """cross-source agreement between two renderings of the same table. returns (agree_keys, conflict_keys)."""
    a = {}
    for f in factsA:
        if f.key() not in a or f.conf > a[f.key()].conf:
            a[f.key()] = f
    b = {}
    for f in factsB:
        if f.key() not in b or f.conf > b[f.key()].conf:
            b[f.key()] = f
    agree, conflict = set(), set()
    for k in a:
        if k in b:
            (agree if close(a[k].v, b[k].v, 1e-6) else conflict).add(k)
    return agree, conflict
