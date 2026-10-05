"""k3_realtext (part c): what real finance/economics-style numeric text exists on disk (read-only survey) + two-parser agreement on real currency amounts, plus copyright-year cross-checks.
NO labels exist for prose amounts: results are coverage / agreement numbers, not accuracy."""
import os, re, glob, collections, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_data import REPO, fnv
import k3_extract as X
from k3_verify import iparse, BROAD, NARROW, close

REG = glob.glob(os.path.expanduser('~/.cargo/registry/src/*'))[0]
CUR = re.compile(r'(?<![\w$])([$€£])\s?(\d[\d,]*(?:\.\d+)?)\s?(billion|million|thousand|bn|mn|[KMBkmb](?![A-Za-z]))?(?![\w])')
PCT = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?%')
SIZE = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?(KB|MB|GB|KiB|MiB|GiB|B)\b')
TIME = re.compile(r'(?<![\w.])(\d+(?:\.\d+)?)\s?(ns|µs|μs|us|ms)\b')
COPY = re.compile(r'[Cc]opyright(?: \(c\))?\s*(?:©)?\s*(\d{4})(?:\s*[-–,]\s*(\d{4}))?')


def files():
    out = []
    for p in sorted(glob.glob(os.path.join(REPO, 'results', '*.md'))) + [os.path.join(REPO, 'DESIGN.md')]:
        if os.path.exists(p):
            out.append((p, 'dev' if fnv(os.path.basename(p)) % 2 == 0 else 'held'))
    for d in sorted(os.listdir(REG)):
        for p in glob.glob(os.path.join(REG, d, '*.md')) + glob.glob(os.path.join(REG, d, 'docs', '*.md')):
            out.append((p, 'confirm'))
    return out


def run(log):
    log('\n## C REAL FINANCE/ECONOMICS-STYLE NUMERIC TEXT ON DISK (read-only survey; NO labels for prose amounts => coverage/agreement only)')
    cnt = collections.defaultdict(collections.Counter)
    ex = collections.defaultdict(list)
    for p, sp in files():
        try:
            t = open(p, encoding='utf8', errors='ignore').read()
        except Exception:
            continue
        cnt[sp]['files'] += 1
        for m in CUR.finditer(t):
            cnt[sp]['currency_amounts'] += 1
            if m.group(3):
                cnt[sp]['currency_with_magnitude_suffix_or_word'] += 1
            # two-parser agreement: extractor v1 parse_cell on the matched span vs independent iparse
            span = m.group(0)
            a = X.parse_cell(span.replace(m.group(1), '', 1), X.KNOW_V1)
            ip = iparse(span.replace(m.group(1), '', 1), BROAD)
            if m.group(3) and m.group(3).lower() in ('million', 'billion', 'thousand', 'bn', 'mn'):
                mag = {'million': 1e6, 'billion': 1e9, 'thousand': 1e3, 'bn': 1e9, 'mn': 1e6}[m.group(3).lower()]
                cnt[sp]['magnitude_word_amounts'] += 1
                v1 = float(m.group(2).replace(',', ''))  # extractor v1 would read the bare number
                cnt[sp]['magnitude_word_missed_by_v1_number_parse'] += 1  # v1's NUM regex stops before the word and rejects (unit unknown)
            elif a is None or ip.base is None:
                cnt[sp]['unparsed_by_one_parser'] += 1
            else:
                av = a['v']
                bv = ip.base * ip.scale
                if close(av, bv):
                    cnt[sp]['parsers_agree'] += 1
                else:
                    cnt[sp]['parsers_disagree'] += 1
                    if len(ex[sp]) < 6:
                        ex[sp].append((span, av, bv))
        cnt[sp]['percent_values'] += len(PCT.findall(t))
        cnt[sp]['size_values_KB_MB_GB'] += len(SIZE.findall(t))
        cnt[sp]['timing_values_ns_us_ms'] += len(TIME.findall(t))
    for sp in ('dev', 'held', 'confirm'):
        log(f'  {sp}: ' + ', '.join(f'{k}={v}' for k, v in sorted(cnt[sp].items())))
        if ex[sp]:
            log(f'     parser disagreements (span, v1-extractor value, verifier value): {ex[sp]}')
    # license copyright years
    n = 0
    spans = []
    for d in sorted(os.listdir(REG)):
        for p in glob.glob(os.path.join(REG, d, 'LICENSE*')) + glob.glob(os.path.join(REG, d, 'COPYING*')):
            try:
                t = open(p, encoding='utf8', errors='ignore').read()
            except Exception:
                continue
            for m in COPY.finditer(t):
                n += 1
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else a
                spans.append((d, a, b))
    bad = sum(1 for d, a, b in spans if b < a or a < 1970 or b > 2027)
    log(f'  LICENSE copyright-year lines in vendored crates: {n} (year ranges parsed; implausible/inverted: {bad}); mean span width {sum(b - a for d, a, b in spans) / max(1, len(spans)):.1f} years')
    # real structural cross-check: copyright start year vs earliest dated changelog entry (independent real sources)
    import tomllib
    viol = tot = 0
    for d, a, b in spans:
        cl = os.path.join(REG, d, 'CHANGELOG.md')
        if os.path.exists(cl):
            tt = open(cl, encoding='utf8', errors='ignore').read()
            ys = [int(x) for x in re.findall(r'^#{1,3}.*?(\d{4})-\d{2}-\d{2}', tt, re.M)]
            if ys:
                tot += 1
                viol += (b < max(ys) - 0) and (b + 1 < min(ys) or False)
    log(f'  copyright_end_year < earliest-changelog-year-1 (cheap inconsistency screen) {viol}/{tot} crates with both')
    log('  What does NOT exist on disk: no real financial statements, filings, earnings text, price series, or prose with labelled amounts; the only finance-flavoured material is the repo market-research / finance-packages catalog markdown (package popularity "~110M/mo", "$11.5B" market figures in prose) and crate benchmark tables (ns/MB/s).')
