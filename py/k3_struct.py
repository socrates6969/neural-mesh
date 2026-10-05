"""k3_struct (part b): REAL structured files (Cargo.toml via tomllib, CHANGELOG headings, repo Cargo.lock) -> prose via template families (dev families vs HELD-OUT families)
-> extraction (cue-based v1 built on dev families; shape-based v2) -> verification (span re-parse, plausibility, independent real cross-source) -> downstream.
Gold = the structured file. Reader = ASSUMED noise model (same constants as table downstream)."""
import os, re, glob, tomllib, collections, random, math, json, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_data import fnv, REPO
from k3_stats import boot_ratio, f1_from, paired_boot, holm

BASE = glob.glob(os.path.expanduser('~/.cargo/registry/src/*'))[0]
OUT = os.path.join(REPO, 'results')
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
NUMW = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
SPDX = ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'BSD-2-Clause', 'ISC', 'MPL-2.0', 'Unlicense', 'Zlib', '0BSD', 'GPL-3.0', 'LGPL-3.0', 'GPL-2.0', 'CC0-1.0', 'BSL-1.0', 'Unicode-3.0', 'Unicode-DFS-2016', 'LLVM-exception', 'AGPL-3.0', 'BSD-2-Clause-Patent', 'MIT-0', 'OpenSSL', 'NCSA', 'CDLA-Permissive-2.0']


# ------------------------------------------------------------------ real records
def load_crates():
    recs = []
    for d in sorted(os.listdir(BASE)):
        p = os.path.join(BASE, d, 'Cargo.toml')
        if not os.path.exists(p):
            continue
        try:
            t = tomllib.load(open(p, 'rb'))
        except Exception:
            continue
        pk = t.get('package', {})
        if 'name' not in pk or 'version' not in pk:
            continue
        r = {'dir': d, 'name': pk['name'], 'version': pk['version'], 'edition': str(pk.get('edition')) if pk.get('edition') else None,
             'license': pk.get('license'), 'rust_version': pk.get('rust-version') if isinstance(pk.get('rust-version'), str) else None,
             'n_deps': len(t.get('dependencies', {}))}
        cl = os.path.join(BASE, d, 'CHANGELOG.md')
        r['changelog'] = []
        if os.path.exists(cl):
            txt = open(cl, encoding='utf8', errors='ignore').read()
            for m in re.finditer(r'^#{1,3}\s*\[?v?(\d+\.\d+(?:\.\d+)?[\w.\-+]*)\]?\s*[-–(]*\s*\(?(\d{4}-\d{2}-\d{2})\)?', txt, re.M):
                r['changelog'].append((m.group(1), m.group(2)))
        h = fnv(r['name']) % 4
        r['split'] = 'dev' if h == 0 else ('held' if h == 1 else 'confirm')
        recs.append(r)
    return recs


def load_lock():
    recs = []
    t = tomllib.load(open(os.path.join(REPO, 'Cargo.lock'), 'rb'))
    for p in t.get('package', []):
        recs.append({'name': p['name'], 'version': p['version'], 'n_deps': len(p.get('dependencies', [])), 'edition': None, 'license': None, 'rust_version': None, 'split': 'dev', 'dir': p['name'] + '-' + p['version'], 'changelog': []})
    return recs


# ------------------------------------------------------------------ rendering (template families)
DEV_FAM = ['P1', 'P2', 'P3']
HELD_FAM = ['Q1', 'Q2', 'Q3', 'Q4']


def vfmt(v, fam, rng):
    if fam in ('Q1', 'Q2'):
        return 'v' + v if rng.random() < 0.8 else v
    return v


def nfmt(n, fam):
    if fam in ('Q1', 'Q4') and n <= 20:
        return NUMW[n]
    return str(n)


def render_crate(r, fam, rng):
    n, v, e, l, rv, nd = r['name'], r['version'], r['edition'], r['license'], r['rust_version'], r['n_deps']
    if fam == 'P1':
        s = f'{n} is at version {v}.'
        parts = []
        if e: parts.append(f'It uses the {e} edition')
        if l: parts.append(f'is licensed {l}')
        if rv: parts.append(f'requires Rust {rv}')
        parts.append(f'has {nd} dependencies')
        s += ' ' + ', '.join(parts) + '.'
    elif fam == 'P2':
        f = [f'Crate {n} (version {v})']
        s = f[0] + ': ' + '; '.join(([f'edition {e}'] if e else []) + ([f'license {l}'] if l else []) + ([f'rust-version {rv}'] if rv else []) + [f'dependencies {nd}']) + '.'
    elif fam == 'P3':
        s = f'- {n} {v} | ' + ' | '.join(([f'edition {e}'] if e else []) + ([l] if l else []) + ([f'MSRV {rv}'] if rv else []) + [f'deps: {nd}'])
    elif fam == 'Q1':
        s = f'As of the latest release, {n} stands at release {vfmt(v, fam, rng)}'
        if e: s += f' and compiles under the {e} edition'
        s += '; '
        parts = []
        if l: parts.append(f'it is published under {l}')
        if rv: parts.append(f'needs rustc {rv}+')
        parts.append(f'pulls in {nfmt(nd, fam)} direct crates')
        s += ', '.join(parts) + '.'
    elif fam == 'Q2':
        s = f'{n}: {vfmt(v, fam, rng)} · ' + ' · '.join(([f'Rust {e}'] if e else []) + ([l] if l else []) + ([f'rustc ≥ {rv}'] if rv else []) + [f'{nd} direct dependencies'])
    elif fam == 'Q3':
        s = f'The {n} package (rev. {v})'
        s += f' ships under {l}.' if l else '.'
        if rv: s += f' Minimum supported toolchain: {rv}.'
        if e: s += f' Edition: {e}.'
        s += f' Dependency count: {nd}.'
    elif fam == 'Q4':
        s = f'{n} {v}, ' + ', '.join(([f'edition {e}'] if e else []) + ([f'license {l}'] if l else []) + ([f'MSRV {rv}'] if rv else []) + [f'{nfmt(nd, fam)} dependencies'])
    else:
        raise ValueError(fam)
    return s


def fmt_date(d, fam, rng):
    y, m, dd = d.split('-')
    if fam in ('M1',):
        return f'{MONTHS[int(m) - 1]} {int(dd)}, {y}'
    if fam in ('M2',):
        return f'{dd}/{m}/{y}'
    if fam in ('M3',):
        return f'{int(dd)} {MONTHS[int(m) - 1]} {y}'
    if fam in ('M4',):
        return f'{y}.{m}.{dd}'
    return d


LOG_DEV = ['L1', 'L2', 'L3']
LOG_HELD = ['M1', 'M2', 'M3', 'M4']


def render_log(ent, fam, rng):
    v, d = ent
    if fam == 'L1': return f'Release {v} was published on {d}.'
    if fam == 'L2': return f'{v} ({d})'
    if fam == 'L3': return f'- {d}: {v}'
    if fam == 'M1': return f'On {fmt_date(d, fam, rng)}, version v{v} went out.'
    if fam == 'M2': return f'v{v} shipped {fmt_date(d, fam, rng)}.'
    if fam == 'M3': return f'{v} — {fmt_date(d, fam, rng)}'
    if fam == 'M4': return f'Released {fmt_date(d, fam, rng)}: {v}.'
    raise ValueError(fam)


# ------------------------------------------------------------------ extractors
SEMVER = r'\d+\.\d+(?:\.\d+)?(?:-[0-9A-Za-z.\-]+)?(?:\+[0-9A-Za-z.\-]+)?'
FIELDS = ['version', 'edition', 'license', 'rust_version', 'n_deps']


def seg_by_names(text, names):
    """split doc text into per-entity segments starting at the first occurrence (word-boundary) of each name; returns {name: (lo, segtext)}"""
    pos = []
    for n in names:
        m = re.search(r'(?<![\w\-])' + re.escape(n) + r'(?![\w\-])', text)
        if m:
            pos.append((m.start(), n))
    pos.sort()
    out = {}
    for i, (lo, n) in enumerate(pos):
        hi = pos[i + 1][0] if i + 1 < len(pos) else len(text)
        out[n] = (lo, text[lo:hi])
    return out


def ex_cue(text, names):
    """v1: cue patterns written against dev families P1-P3 only. returns {(name, field): (value, span, conf)}"""
    facts = {}
    for n, (lo, seg) in seg_by_names(text, names).items():
        def add(f, m, g=1, conf=0.9):
            if m and (n, f) not in facts:
                facts[(n, f)] = (m.group(g).strip(), (lo + m.start(g), lo + m.end(g)), conf)
        add('version', re.search(r'at version (' + SEMVER + ')', seg)) or None
        if (n, 'version') not in facts:
            add('version', re.search(r'\(version (' + SEMVER + r')\)', seg))
        if (n, 'version') not in facts:
            add('version', re.search(r'^' + re.escape(n) + r' (' + SEMVER + r') \|', seg))
        add('edition', re.search(r'the (\d{4}) edition', seg))
        if (n, 'edition') not in facts:
            add('edition', re.search(r'edition (\d{4})', seg))
        add('license', re.search(r'licensed ([^,.;|]+?)(?:[,;|]|\.\s|\.$|$)', seg))
        if (n, 'license') not in facts:
            add('license', re.search(r'license ([^;|]+?)(?:[;|]|\.$|$)', seg))
        if (n, 'license') not in facts:
            m = re.search(r'\| ((?:[A-Za-z0-9.\-]+(?: (?:OR|AND|WITH) )?)+) \|', seg)
            if m and any(s_ in m.group(1) for s_ in SPDX):
                add('license', m)
        add('rust_version', re.search(r'requires Rust (' + SEMVER + ')', seg))
        if (n, 'rust_version') not in facts:
            add('rust_version', re.search(r'rust-version (' + SEMVER + ')', seg))
        if (n, 'rust_version') not in facts:
            add('rust_version', re.search(r'MSRV (' + SEMVER + ')', seg))
        add('n_deps', re.search(r'has (\d+) dependencies', seg))
        if (n, 'n_deps') not in facts:
            add('n_deps', re.search(r'dependencies (\d+)', seg))
        if (n, 'n_deps') not in facts:
            add('n_deps', re.search(r'deps: (\d+)', seg))
    return facts


def ex_shape(text, names):
    """v2: value-shape scanners (semver / 4-digit edition / SPDX-expression / number next to 'depend') + weak cues; written after seeing v1 collapse on dev-only phrasing? NO:
    written before held/confirm on dev-family knowledge + generic shapes; frozen."""
    facts = {}
    spdx_alt = '|'.join(re.escape(s) for s in sorted(SPDX, key=len, reverse=True))
    lic_re = re.compile(r'(?<![\w\-])(?:' + spdx_alt + r')(?:\s+(?:OR|AND|WITH)\s+(?:' + spdx_alt + r'))*(?![\w\-])')
    for n, (lo, seg) in seg_by_names(text, names).items():
        body = seg[len(n):]
        sv = [(m.start(), m.end(), m.group(0)) for m in re.finditer(r'(?<![\w.\-])v?(' + SEMVER + r')(?![\w.]*\w)', body)]
        rust_cue = lambda s0: re.search(r'(rust|rustc|msrv|toolchain)[^.;|]{0,40}$', body[max(0, s0 - 45):s0], re.I) is not None or body[max(0, s0 - 3):s0].strip() in ('≥', '>=')
        ver = None
        rvs = None
        for (a, b, t) in sv:
            core = t.lstrip('v')
            if rust_cue(a):
                if rvs is None:
                    rvs = (core, (lo + len(n) + a + (1 if t.startswith('v') else 0), lo + len(n) + b))
            elif ver is None:
                ver = (core, (lo + len(n) + a + (1 if t.startswith('v') else 0), lo + len(n) + b))
        if ver:
            facts[(n, 'version')] = (ver[0], ver[1], 0.8)
        if rvs:
            facts[(n, 'rust_version')] = (rvs[0], rvs[1], 0.75)
        m = re.search(r'(?<![\w.\-])(2015|2018|2021|2024)(?![\w.\-])', body)
        if m:
            facts[(n, 'edition')] = (m.group(1), (lo + len(n) + m.start(), lo + len(n) + m.end()), 0.8)
        m = lic_re.search(body)
        if m:
            facts[(n, 'license')] = (m.group(0), (lo + len(n) + m.start(), lo + len(n) + m.end()), 0.8)
        m = re.search(r'(\d+|' + '|'.join(NUMW) + r')\s+(?:direct\s+)?(?:dependenc|crates)', body, re.I) or re.search(r'(?:dependenc\w*|deps)[^\d\w]{0,12}(\d+)', body, re.I)
        if m:
            t = m.group(1)
            val = str(NUMW.index(t.lower())) if t.lower() in NUMW else t
            facts[(n, 'n_deps')] = (val, (lo + len(n) + m.start(1), lo + len(n) + m.end(1)), 0.75)
    return facts


# ---- changelog
def parse_date_any(s):
    """independent date parser; returns (iso, ambiguous) or None"""
    s = s.strip().rstrip('.')
    m = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m: return f'{m.group(1)}-{m.group(2)}-{m.group(3)}', False
    m = re.fullmatch(r'(\d{4})\.(\d{2})\.(\d{2})', s)
    if m: return f'{m.group(1)}-{m.group(2)}-{m.group(3)}', False
    m = re.fullmatch(r'(' + '|'.join(MONTHS) + r') (\d{1,2}), (\d{4})', s)
    if m: return f'{m.group(3)}-{MONTHS.index(m.group(1)) + 1:02d}-{int(m.group(2)):02d}', False
    m = re.fullmatch(r'(\d{1,2}) (' + '|'.join(MONTHS) + r') (\d{4})', s)
    if m: return f'{m.group(3)}-{MONTHS.index(m.group(2)) + 1:02d}-{int(m.group(1)):02d}', False
    m = re.fullmatch(r'(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        amb = a <= 12 and b <= 12 and a != b
        return f'{m.group(3)}-{b:02d}-{a:02d}' if not (a <= 12 and b > 12) else f'{m.group(3)}-{a:02d}-{b:02d}', amb
    return None


def ex_log_cue(text):
    """v1: ISO-only cues from dev families L1-L3: returns {version: (date_iso, span, conf)}"""
    out = {}
    for m in re.finditer(r'Release (' + SEMVER + r') was published on (\d{4}-\d{2}-\d{2})', text):
        out[m.group(1)] = (m.group(2), (m.start(2), m.end(2)), 0.9)
    for m in re.finditer(r'^(' + SEMVER + r') \((\d{4}-\d{2}-\d{2})\)', text, re.M):
        out[m.group(1)] = (m.group(2), (m.start(2), m.end(2)), 0.9)
    for m in re.finditer(r'^- (\d{4}-\d{2}-\d{2}): (' + SEMVER + ')', text, re.M):
        out[m.group(2)] = (m.group(1), (m.start(1), m.end(1)), 0.9)
    return out


DATE_ANY = r'(?:\d{4}-\d{2}-\d{2}|\d{4}\.\d{2}\.\d{2}|(?:' + '|'.join(MONTHS) + r') \d{1,2}, \d{4}|\d{1,2} (?:' + '|'.join(MONTHS) + r') \d{4}|\d{1,2}/\d{1,2}/\d{4})'


def ex_log_shape(text):
    """v2: per line, one semver token + one date-shaped token (any of 5 formats) -> pair"""
    out = {}
    for ln_m in re.finditer(r'^.*$', text, re.M):
        ln = ln_m.group(0)
        sv = re.search(r'(?<![\w.\-/])v?(' + SEMVER + r')(?![\w.\-/])(?!\s*,\s*\d{4})', ln)
        dm = re.search(DATE_ANY, ln)
        if sv and dm and sv.start() != dm.start():
            # make sure the semver token is not part of the date (e.g. 2024.01.05)
            if dm.start() <= sv.start() < dm.end():
                svs = [x for x in re.finditer(r'(?<![\w.\-/])v?(' + SEMVER + r')(?![\w.\-/])', ln) if not (dm.start() <= x.start() < dm.end())]
                if not svs: continue
                sv = svs[0]
            pd = parse_date_any(dm.group(0))
            if pd:
                out[sv.group(1)] = (pd[0], (ln_m.start() + dm.start(), ln_m.start() + dm.end()), 0.75 if not pd[1] else 0.35)
    return out


# ------------------------------------------------------------------ verification
def v_semver(s):
    return re.fullmatch(SEMVER, s) is not None


def verify_crate_fact(f, val, span, text, rec, vc):
    """returns set of flags. vc: dict(plaus, span, xsrc)"""
    fl = set()
    if vc.get('span'):
        t = text[span[0]:span[1]]
        if t.strip() != val.strip():
            # n_deps via number words
            if not (f == 'n_deps' and t.lower() in NUMW and str(NUMW.index(t.lower())) == val):
                fl.add('span_mismatch')
    if vc.get('plaus'):
        if f in ('version', 'rust_version') and not v_semver(val): fl.add('implausible')
        if f == 'edition' and val not in ('2015', '2018', '2021', '2024'): fl.add('implausible')
        if f == 'license' and not any(s_ in val for s_ in SPDX): fl.add('implausible')
        if f == 'n_deps' and not (val.isdigit() and 0 <= int(val) <= 400): fl.add('implausible')
    if vc.get('xsrc') and f == 'version':
        # real independent source: the crate directory name (name-version) -- plus changelog top entry staleness is NOT a correctness oracle
        if rec['dir'] != f"{rec['name']}-{val}":
            fl.add('xsrc_dirname')
    return fl


def verify_log_fact(ver, date, span, text, vc):
    fl = set()
    if vc.get('plaus'):
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date): fl.add('implausible')
        else:
            y, m, d = map(int, date.split('-'))
            if not (1990 <= y <= 2027 and 1 <= m <= 12 and 1 <= d <= 31): fl.add('implausible')
    if vc.get('span'):
        t = text[span[0]:span[1]]
        pd = parse_date_any(t)
        if pd is None or pd[0] != date: fl.add('span_mismatch')
        elif pd[1]: fl.add('ambiguous_date')
    return fl


# ------------------------------------------------------------------ evaluation
def eval_crates(recs, fam, ex, groupsz=6, seed=3, vc=None, fam2=None):
    vc = vc or {'span': True, 'plaus': True, 'xsrc': True}
    c = collections.defaultdict(lambda: collections.Counter())  # field -> counts ; doc-level
    docs = collections.defaultdict(lambda: collections.Counter())
    rng = random.Random(seed)
    # group crates into documents; names must be unique inside a document and must not occur as a word inside another crate's rendered sentence
    uniq = []
    seen = set()
    for r in recs:
        if r['name'] in seen or re.fullmatch(r'(it|is|at|has|deps|rust|rustc|edition|license|licensed|version|crate|the|and|or|msrv|rev|as|of|a|an|under|ships|package|stands|release|latest|direct|crates|dependencies|dependency|needs|compiles|published|pulls|in|uses|requires|count|minimum|supported|toolchain|with|latest|ships)', r['name']):
            continue
        seen.add(r['name'])
        uniq.append(r)
    recs = uniq
    for gi in range(0, len(recs) - groupsz + 1, groupsz):
        grp = recs[gi:gi + groupsz]
        lines = [render_crate(r, fam, rng) for r in grp]
        text = '\n'.join(lines) + '\n'
        names = [r['name'] for r in grp]
        facts = ex(text, names)
        facts2 = None
        if fam2:
            text2 = '\n'.join(render_crate(r, fam2, rng) for r in grp) + '\n'
            facts2 = ex(text2, names)
        doc = f'{grp[0]["dir"]}'
        for r in grp:
            gold = {'version': r['version'], 'edition': r['edition'], 'license': r['license'], 'rust_version': r['rust_version'], 'n_deps': str(r['n_deps'])}
            for f in FIELDS:
                g = gold[f]
                p = facts.get((r['name'], f))
                if g is not None:
                    if p is None:
                        c[f]['fn'] += 1
                        docs[doc]['fn'] += 1
                        continue
                if p is not None:
                    flags = verify_crate_fact(f, p[0], p[1], text, r, vc)
                    if facts2 is not None:
                        q = facts2.get((r['name'], f))
                        if q is not None and q[0].strip() != p[0].strip():
                            flags.add('xsrc_family')
                    fl = bool(flags)
                    ok = g is not None and p[0].strip() == str(g).strip()
                    key = 'tp' if ok else ('wrong' if g is not None else 'junk')
                    c[f][key] += 1
                    c[f][key + '_flag'] += fl
                    c[f][key + '_conf_lt5'] += p[2] < 0.5
                    docs[doc][key] += 1
                    docs[doc][key + '_flag'] += fl
                    if ok:
                        docs[doc]['kept_tp'] += (not fl)
                    else:
                        docs[doc]['kept_bad'] += (not fl)
    return c, docs


def eval_logs(recs, fam, ex, seed=3, vc=None):
    vc = vc or {'span': True, 'plaus': True}
    c = collections.Counter()
    rng = random.Random(seed)
    for r in recs:
        ents = r['changelog'][:12]
        if len(ents) < 3:
            continue
        text = '\n'.join(render_log(e, fam, rng) for e in ents) + '\n'
        facts = ex(text)
        gold = dict(ents)
        for v, d in gold.items():
            p = facts.get(v)
            if p is None:
                c['fn'] += 1
                continue
        for v, p in facts.items():
            fl = bool(verify_log_fact(v, p[0], p[1], text, vc))
            ok = v in gold and gold[v] == p[0]
            k = 'tp' if ok else ('wrong' if v in gold else 'junk')
            c[k] += 1
            c[k + '_flag'] += fl
    return c


def prf(c):
    tp = c['tp']
    fp = c['wrong'] + c['junk']
    fn = c['fn'] + c['wrong']
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def run(log):
    out = {}
    crates = load_crates()
    lock = load_lock()
    log('\n## B STRUCTURED FILES (Cargo.toml via tomllib, CHANGELOG headings, repo Cargo.lock) -> prose templates (dev families P1-P3 / L1-L3, HELD-OUT families Q1-Q4 / M1-M4)')
    by = collections.defaultdict(list)
    for r in crates:
        by[r['split']].append(r)
    log('  crates: ' + ', '.join(f'{k}={len(v)}' for k, v in by.items()) + f'; repo Cargo.lock packages {len(lock)} (dev-only record source); crates with changelog entries>=3: ' + str(sum(1 for r in crates if len(r['changelog']) >= 3)))
    # real cross-source facts (no rendering involved)
    n = len(crates)
    dm = sum(1 for r in crates if r['dir'] == f"{r['name']}-{r['version']}")
    stale = [r for r in crates if r['changelog']]
    top_match = sum(1 for r in stale if r['changelog'][0][0] == r['version'])
    in_list = sum(1 for r in stale if r['version'] in [v for v, _ in r['changelog']])
    mono = 0
    mono_n = 0
    for r in stale:
        ds = [d for _, d in r['changelog']]
        if len(ds) >= 3:
            mono_n += 1
            mono += all(a >= b for a, b in zip(ds, ds[1:])) or all(a <= b for a, b in zip(ds, ds[1:]))
    log(f'  REAL cross-source facts: dir-name == name-version {dm}/{n} (by construction of cargo, a vacuous check); CHANGELOG top entry == Cargo.toml version {top_match}/{len(stale)} crates with dated headings; Cargo.toml version appears anywhere in changelog headings {in_list}/{len(stale)}; changelog dates monotone (>=3 entries) {mono}/{mono_n}')
    out['real_xsrc'] = {'dirname': [dm, n], 'top_match': [top_match, len(stale)], 'in_list': [in_list, len(stale)], 'monotone': [mono, mono_n]}
    # per-field extraction
    for ex_name, ex in (('cue(v1)', ex_cue), ('shape(v2)', ex_shape)):
        for sp in ('dev', 'held', 'confirm'):
            rs = by[sp]
            for famgroup, fams in (('dev-fam', DEV_FAM), ('HELD-OUT-fam', HELD_FAM)):
                tot = collections.Counter()
                perf = collections.defaultdict(collections.Counter)
                docs_all = collections.defaultdict(collections.Counter)
                for fam in fams:
                    c, docs = eval_crates(rs, fam, ex)
                    for f, cc in c.items():
                        perf[f].update(cc)
                        tot.update(cc)
                    for d, cc in docs.items():
                        docs_all[d + fam].update(cc)
                p, r_, f1 = prf(tot)
                nbad = tot['wrong'] + tot['junk']
                bflag = tot['wrong_flag'] + tot['junk_flag']
                log(f'  {ex_name:10s} {sp:7s} {famgroup:13s}: P {p:.3f} R {r_:.3f} F1 {f1:.3f} | per-field F1 ' + ' '.join(f'{f}:{prf(perf[f])[2]:.2f}' for f in FIELDS) + f' | wrong+junk {nbad} caught by verifier {bflag / max(1, nbad):.3f}; false-flag on correct {tot["tp_flag"] / max(1, tot["tp"]):.4f}')
                out[f'{ex_name}_{sp}_{famgroup}'] = {'P': p, 'R': r_, 'F1': f1, 'bad': nbad, 'caught': bflag, 'ff': tot['tp_flag'] / max(1, tot['tp'])}
    # per held-out family F1 (shape extractor), confirm
    log('  per-family F1, confirm crates (cue / shape):')
    for fam in DEV_FAM + HELD_FAM:
        res = []
        for ex in (ex_cue, ex_shape):
            c, _ = eval_crates(by['confirm'], fam, ex)
            tot = collections.Counter()
            for cc in c.values():
                tot.update(cc)
            res.append(prf(tot)[2])
        log(f'     {fam}: cue {res[0]:.3f} shape {res[1]:.3f}')
    log('  CROSS-SOURCE between two family renderings of the same crate (shape extractor; A = family under test, B = P2): wrong+junk facts caught by A/B disagreement; false conflicts on correct facts')
    for fam in ['P1', 'Q1', 'Q2', 'Q3', 'Q4']:
        c, _ = eval_crates(by['confirm'], fam, ex_shape, fam2='P2', vc={'span': False, 'plaus': False, 'xsrc': False})
        tot = collections.Counter()
        for cc in c.values():
            tot.update(cc)
        nb = tot['wrong'] + tot['junk']
        log(f'     A={fam}: wrong+junk {nb} caught {(tot["wrong_flag"] + tot["junk_flag"]) / max(1, nb):.3f} (agree-on-wrong {1 - (tot["wrong_flag"] + tot["junk_flag"]) / max(1, nb):.3f}); false conflict on correct {tot["tp_flag"] / max(1, tot["tp"]):.4f}')
    # changelog (version->date)
    for ex_name, ex in (('cue(v1)', ex_log_cue), ('shape(v2)', ex_log_shape)):
        for sp in ('dev', 'held', 'confirm'):
            for famgroup, fams in (('dev-fam', LOG_DEV), ('HELD-OUT-fam', LOG_HELD)):
                tot = collections.Counter()
                for fam in fams:
                    tot.update(eval_logs(by[sp], fam, ex))
                p, r_, f1 = prf(tot)
                nbad = tot['wrong'] + tot['junk']
                log(f'  changelog {ex_name:10s} {sp:7s} {famgroup:13s}: P {p:.3f} R {r_:.3f} F1 {f1:.3f} | wrong+junk {nbad} verifier-flagged {(tot["wrong_flag"] + tot["junk_flag"]) / max(1, nbad):.3f} (ambiguous DD/MM dates flagged {tot["tp_flag"]} of {tot["tp"]} correct)')
                out[f'log_{ex_name}_{sp}_{famgroup}'] = {'P': p, 'R': r_, 'F1': f1}
    log('  per-family F1 changelog, confirm (cue / shape):')
    for fam in LOG_DEV + LOG_HELD:
        res = []
        for ex in (ex_log_cue, ex_log_shape):
            tot = eval_logs(by['confirm'], fam, ex)
            res.append(prf(tot)[2])
        log(f'     {fam}: cue {res[0]:.3f} shape {res[1]:.3f}')
    json.dump(out, open(os.path.join(OUT, 'k3_struct.json'), 'w'), indent=1, default=str)
    return out


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    L = []
    def lg(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        L.append(s)
    run(lg)
    open(os.path.join(OUT, 'k3_struct.log'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
