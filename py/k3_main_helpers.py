"""shared imports/helpers for k3_main_* modules (avoids circular import with k3_main)"""
import sys, os, collections, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k3_data import load_all, usable, corpus_hash, REPO
from k3_render import render_table, LAYOUTS, VFMTS
import k3_extract as X
from k3_eval import run_grid, score_render, gclass, best_by_key, close, SEED
from k3_verify import VFULL, VNARROW, VCfg, verify_facts, is_flagged, HARD_FLAGS, crosscheck, footing_applicable
from k3_stats import boot_ratio, f1_from, paired_boot, holm

KNOWS = {'v1': X.KNOW_V1, 'v2': X.KNOW_V2, 'v3': X.KNOW_V3}
OUT = os.path.join(REPO, 'results')


def inc(rc, tau=0.0, vname=None, repair=False, strict=True, flagset=HARD_FLAGS):
    ok = rc['strict'] if strict else rc['lenient']
    if rc['kind'] == 'fn':
        return (0, 0, 1)
    if rc['conf'] < tau:
        return (0, 0, 1) if rc['kind'] in ('tp', 'wrong') else (0, 0, 0)
    if vname and is_flagged(rc['flags'][vname], flagset):
        if repair and rc['rep_ok'][vname]:
            return (1, 0, 0)
        return (0, 0, 1) if rc['kind'] in ('tp', 'wrong') else (0, 0, 0)
    if rc['kind'] == 'junk':
        return (0, 1, 0)
    if ok:
        return (1, 0, 0)
    return (0, 1, 1)


def doc_counts(renders, keyfn, **kw):
    G = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for rd in renders:
        k = keyfn(rd)
        v = G[k][rd['doc']]
        for rc in rd['recs']:
            a = inc(rc, **kw)
            v[0] += a[0]
            v[1] += a[1]
            v[2] += a[2]
    return G


def summarize(G, B=400):
    out = {}
    for k, docs in G.items():
        pt, lo, hi = boot_ratio(docs, f1_from, B=B)
        tot = [sum(v[i] for v in docs.values()) for i in range(3)]
        p = tot[0] / (tot[0] + tot[1]) if tot[0] + tot[1] else 0.0
        r = tot[0] / (tot[0] + tot[2]) if tot[0] + tot[2] else 0.0
        out[k] = {'f1': pt, 'lo': lo, 'hi': hi, 'P': p, 'R': r, 'tp': tot[0], 'fp': tot[1], 'fn': tot[2], 'ndoc': len(docs)}
    return out
