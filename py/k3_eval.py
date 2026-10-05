"""k3_eval: render -> extract -> score vs GOLD (real pipe grid) -> verify. Produces per-fact records; summaries in k3_report.py."""
import sys, re, math, random, time, pickle, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_data import load_all, usable, corpus_hash
from k3_render import render_table, LAYOUTS, VFMTS
from k3_extract import extract, KNOW_V1, KNOW_V2
from k3_verify import verify_facts, VFULL, VNARROW, VCfg, crosscheck, is_flagged, close, iparse, footing_applicable, HARD_FLAGS

SEED = 17


def gclass(g):
    u = g['unit']
    raw = g['raw']
    if u is None:
        return 'thousands' if re.search(r'\d,\d{3}', raw) else 'plain'
    if u == '%' or u.startswith('%'):
        return 'pct'
    if u in ('K', 'k', 'M', 'B', 'm', 'b'):
        return 'sfx'
    if u.startswith('hdr:'):
        return 'hdr_unit'
    base = re.sub(r'\s', '', u).lower().split('/')[0]
    if base in ('ns', 'µs', 'μs', 'us', 'ms', 's', 'sec', 'secs', 'min', 'minutes'):
        return 'unit_time'
    if base in ('b', 'kb', 'mb', 'gb', 'kib', 'mib', 'gib', 'bytes', 'byte'):
        return 'unit_size'
    return 'unit_other'


def best_by_key(facts):
    d = {}
    for f in facts:
        k = f.key()
        if k not in d or f.conf > d[k].conf:
            d[k] = f
    return d


def score_render(T, text, know, vcfgs):
    """returns list of records: dict(kind, cls, key, conf, strict, lenient, flags{name:set}, repair_ok{name:bool})"""
    facts = extract(text, know)
    best = best_by_key(facts)
    gold = T['facts']
    ver = {}
    for name, vc in vcfgs.items():
        ver[name] = verify_facts(text, list(best.values()), vc)
    recs = []
    for k, f in best.items():
        g = gold.get(k)
        if g is None:
            kind, cls, strict, lenient = 'junk', 'none', False, False
        else:
            strict = close(f.v, g['value'], 1e-6)
            sgn = -1 if g['aw'] < 0 else 1
            lenient = strict or close(abs(f.base), abs(g['aw']), 1e-6)
            kind = 'tp' if strict else 'wrong'
            cls = gclass(g)
        flags = {n: ver[n][id(f)]['flags'] for n in vcfgs}
        rep_ok = {}
        for n in vcfgs:
            rp = ver[n][id(f)]['repair']
            rep_ok[n] = (rp is not None and g is not None and close(rp, g['value'], 1e-6))
        recs.append({'kind': kind, 'cls': cls, 'key': k, 'conf': f.conf, 'strict': strict, 'lenient': lenient, 'flags': flags, 'rep_ok': rep_ok, 'struct': f.struct, 'notes': f.notes,
                     'repair': {n: ver[n][id(f)]['repair'] for n in vcfgs}})
    for k, g in gold.items():
        if k not in best:
            recs.append({'kind': 'fn', 'cls': gclass(g), 'key': k, 'conf': 0.0, 'strict': False, 'lenient': False, 'flags': {n: set() for n in vcfgs}, 'rep_ok': {n: False for n in vcfgs}, 'struct': None, 'notes': [], 'repair': {}})
    return recs, facts, best


def run_grid(tables, layouts, vfmts, know, vcfgs=None, seed=SEED, progress=False):
    vcfgs = vcfgs or {'full': VFULL, 'narrow': VNARROW}
    out = []
    t0 = time.time()
    n = 0
    for T in tables:
        for lay in layouts:
            for vf in vfmts:
                r = render_table(T, lay, vf, seed)
                if r is None:
                    continue
                recs, facts, best = score_render(T, r['text'], know, vcfgs)
                out.append({'split': T['split'], 'doc': T['doc'], 'tid': T['id'], 'crate': T['crate'], 'layout': lay, 'vfmt': vf, 'recs': recs, 'ngold': len(T['facts']),
                            'ctx_numbers': r['ctx_numbers'], 'nfact': len(best)})
                n += 1
        if progress and n % 200 == 0:
            print('..', n, round(time.time() - t0, 1), file=sys.stderr)
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def agg(renders, key=lambda x: 'ALL', tau=0.0, strict=True, vname=None, repair=False, flagset=HARD_FLAGS):
    """micro P/R/F1 grouped. vname: apply verifier (drop flagged; optional repair)."""
    G = collections.defaultdict(lambda: [0, 0, 0])
    for rd in renders:
        k = key(rd)
        for rc in rd['recs']:
            ok = rc['strict'] if strict else rc['lenient']
            if rc['kind'] == 'fn':
                G[k][2] += 1
                continue
            if rc['conf'] < tau:
                if rc['kind'] in ('tp', 'wrong') and True:
                    G[k][2] += 1
                continue
            if vname and is_flagged(rc['flags'][vname], flagset):
                if repair and rc['rep_ok'][vname]:
                    G[k][0] += 1
                    continue
                if rc['kind'] in ('tp', 'wrong'):
                    G[k][2] += 1
                continue
            if rc['kind'] == 'junk':
                G[k][1] += 1
            elif ok:
                G[k][0] += 1
            else:
                G[k][1] += 1
                G[k][2] += 1
    return {k: prf(*v) + tuple(v) for k, v in G.items()}


def catch_stats(renders, vname, flagset=HARD_FLAGS, strict=True):
    """catch rate of wrong extractions (kind wrong or junk) and false-flag rate on correct ones."""
    c = collections.Counter()
    for rd in renders:
        for rc in rd['recs']:
            if rc['kind'] == 'fn':
                continue
            fl = is_flagged(rc['flags'][vname], flagset)
            correct = rc['kind'] == 'tp'
            if correct:
                c['ok'] += 1
                c['ok_flag'] += fl
            else:
                kk = rc['kind']
                c[kk] += 1
                c[kk + '_flag'] += fl
                if kk == 'wrong':
                    c['wrong_repair_ok'] += (fl and rc['rep_ok'][vname])
    return c


if __name__ == '__main__':
    tables, st = load_all()
    U = [T for T in tables if usable(T)]
    print('corpus fnv', corpus_hash(tables), file=sys.stderr)
    dev = [T for T in U if T['split'] == 'dev']
    t0 = time.time()
    rs = run_grid(dev[:10], ['pipe', 'ws', 'F2'], ['v0', 'mix'], KNOW_V1)
    print('time', time.time() - t0, len(rs))
    a = agg(rs, key=lambda r: (r['layout'], r['vfmt']))
    for k, v in sorted(a.items()):
        print(k, ['%.3f' % x for x in v[:3]], v[3:])
