"""k3_main: experiment driver. Usage: python py/k3_main.py <cmd> [--splits dev,held,confirm]
cmds: corpus | extract | verify | controls | down | all.   All numbers are on REAL table grids from disk with SYNTHETIC renderings / reader-noise model (labelled)."""
import sys, os, json, time, collections, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from k3_data import load_all, usable, corpus_hash, REPO
from k3_render import render_table, LAYOUTS, VFMTS
import k3_extract as X
from k3_eval import run_grid, score_render, gclass, best_by_key, close, SEED
from k3_verify import VFULL, VNARROW, VCfg, verify_facts, is_flagged, HARD_FLAGS, crosscheck, footing_applicable
from k3_stats import boot_ratio, f1_from, paired_boot, holm

KNOWS = {'v1': X.KNOW_V1, 'v2': X.KNOW_V2, 'v3': X.KNOW_V3}
OUT = os.path.join(REPO, 'results')
LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.append(s)


def inc(rc, tau=0.0, vname=None, repair=False, strict=True, flagset=HARD_FLAGS):
    """-> (tp, fp, fn) increments for one record under a decision policy"""
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


def load():
    tables, st = load_all()
    U = [T for T in tables if usable(T)]
    return tables, U


def cmd_corpus(tables, U):
    log('## CORPUS (REAL markdown tables; gold = pipe grid)')
    log('files_with_tables', len({T['doc'] for T in tables}), 'raw_tables', len(tables), 'usable', len(U), 'fnv', corpus_hash(tables))
    for sp in ('dev', 'held', 'confirm'):
        sel = [T for T in U if T['split'] == sp]
        crates = len({T['crate'] for T in sel})
        log(f'  {sp}: tables={len(sel)} docs={len({T["doc"] for T in sel})} crates/repos={crates} gold_numeric_cells={sum(T["nnum"] for T in sel)}')
    cl = collections.Counter()
    for T in U:
        for k, g in T['facts'].items():
            cl[(T['split'], gclass(g))] += 1
    for sp in ('dev', 'held', 'confirm'):
        log('  gold cell classes', sp, dict(sorted((k[1], v) for k, v in cl.items() if k[0] == sp)))
    neg = collections.Counter()
    for T in U:
        neg[T['split']] += sum(1 for g in T['facts'].values() if g['value'] < 0)
    log('  negative gold cells', dict(neg))
    tot = collections.Counter()
    for T in U:
        tot['tables_with_total_row'] += any(r.startswith('total') or r.startswith('sum') or r.startswith('overall') for r in T['rk'])
        tot['tables_with_total_col'] += any(h.startswith('total') or h.startswith('sum') for h in T['hk'])
    log('  footing-eligible tables (total row/col label present):', dict(tot))
    # sample renders
    with open(os.path.join(OUT, 'k3_samples.txt'), 'w', encoding='utf-8') as f:
        T = [t for t in U if t['split'] == 'confirm' and len(t['hk']) >= 4 and t['nnum'] >= 8][3]
        for lay in LAYOUTS:
            for vf in ('v0', 'mix'):
                r = render_table(T, lay, vf, SEED)
                f.write(f'===== {T["id"]} layout={lay} vfmt={vf}\n')
                f.write((r['text'] if r else '(not applicable)') + '\n')


def run_all_grids(U, know, layouts, vfmts, splits):
    out = {}
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        out[sp] = run_grid(sel, layouts, vfmts, know)
    return out


def cmd_extract(tables, U, splits):
    res = {}
    log('\n## E1 extraction F1 per layout (v0 = original value text, mix = value perturbations), micro over cells, 95% CI = bootstrap over documents(files)')
    for kn in ('v1', 'v2', 'v3'):
        grids = run_all_grids(U, KNOWS[kn], LAYOUTS, ['v0', 'mix'], splits)
        res[kn] = {}
        for sp in splits:
            S = summarize(doc_counts(grids[sp], lambda r: (r['layout'], r['vfmt'])))
            Sl = summarize(doc_counts(grids[sp], lambda r: (r['layout'], r['vfmt']), strict=False))
            res[kn][sp] = {f'{k[0]}/{k[1]}': v for k, v in S.items()}
            log(f'-- extractor {kn} split {sp}  (strict value F1 [lenient=number-as-written accepted])')
            for lay in LAYOUTS:
                row = []
                for vf in ('v0', 'mix'):
                    s = S.get((lay, vf))
                    if s is None:
                        row.append('n/a')
                        continue
                    row.append(f'{s["f1"]:.3f} [{s["lo"]:.3f},{s["hi"]:.3f}] (P{s["P"]:.2f}/R{s["R"]:.2f}; lenient {Sl[(lay, vf)]["f1"]:.3f}; cells {s["tp"] + s["fn"]})')
                log(f'   {lay:7s} v0 {row[0]}   |  mix {row[1]}')
    # E2 per value-format on ws/F2/H2/pipe for v3 and v1
    log('\n## E2 extraction F1 per value-format perturbation (layouts pipe, ws, F2 [dev family], H2 [held-out family] pooled per layout)')
    for kn in ('v1', 'v3'):
        grids = run_all_grids(U, KNOWS[kn], ['pipe', 'ws', 'F2', 'H2'], VFMTS, splits)
        res[kn + '_vfmt'] = {}
        for sp in splits:
            S = summarize(doc_counts(grids[sp], lambda r: (r['layout'], r['vfmt'])), B=200)
            res[kn + '_vfmt'][sp] = {f'{k[0]}/{k[1]}': v for k, v in S.items()}
            log(f'-- extractor {kn} split {sp}')
            for lay in ('pipe', 'ws', 'F2', 'H2'):
                log('   %-5s ' % lay + '  '.join(f'{vf}:{S[(lay, vf)]["f1"]:.3f}' for vf in VFMTS if (lay, vf) in S))
        # per gold field class
        for sp in splits:
            G = doc_counts([r for r in grids[sp] if r['layout'] in ('pipe', 'ws', 'F2')], lambda r: 'ALL')
            cls = collections.defaultdict(lambda: [0, 0, 0])
            for rd in grids[sp]:
                if rd['layout'] not in ('pipe', 'ws', 'F2') or rd['vfmt'] not in ('v0', 'mix'):
                    continue
                for rc in rd['recs']:
                    a = inc(rc)
                    c = cls[rc['cls']]
                    c[0] += a[0]
                    c[1] += a[1]
                    c[2] += a[2]
            log(f'   per gold-field-class F1 [{kn} {sp}; pipe+ws+F2, v0+mix]: ' + ', '.join(f'{k}:{f1_from(v):.3f}(n={v[0] + v[2]})' for k, v in sorted(cls.items()) if k != 'none'))
            res[kn + '_class_' + sp] = {k: {'f1': f1_from(v), 'n': v[0] + v[2]} for k, v in cls.items()}
    json.dump(res, open(os.path.join(OUT, 'k3_extract.json'), 'w'), indent=1, default=str)
    return res


def cmd_abstain(tables, U, splits):
    log('\n## E3 abstention: precision/recall vs confidence threshold tau (v3, layouts ws+F2+H2+wrap+split, mix; fraction of extracted facts kept)')
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        rs = run_grid(sel, ['ws', 'wrap', 'split', 'F2', 'H2'], ['mix'], X.KNOW_V3)
        row = []
        for tau in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95):
            S = summarize(doc_counts(rs, lambda r: 'ALL', tau=tau), B=100)['ALL']
            kept = sum(1 for rd in rs for rc in rd['recs'] if rc['kind'] != 'fn' and rc['conf'] >= tau)
            allp = sum(1 for rd in rs for rc in rd['recs'] if rc['kind'] != 'fn')
            row.append(f'tau{tau}: P{S["P"]:.3f} R{S["R"]:.3f} F1 {S["f1"]:.3f} kept {kept / max(1, allp):.2f}')
        log(f'  {sp}: ' + ' | '.join(row))
        # calibration: mean conf of correct vs wrong and ECE (10 bins)
        bins = collections.defaultdict(lambda: [0, 0.0, 0])
        for rd in rs:
            for rc in rd['recs']:
                if rc['kind'] == 'fn':
                    continue
                b = min(9, int(rc['conf'] * 10))
                bins[b][0] += 1
                bins[b][1] += rc['conf']
                bins[b][2] += rc['kind'] == 'tp'
        n = sum(v[0] for v in bins.values())
        ece = sum(abs(v[1] / v[0] - v[2] / v[0]) * v[0] for v in bins.values() if v[0]) / max(1, n)
        log(f'     ECE (confidence vs strict correctness) {ece:.3f} over {n} extracted facts; per-bin acc: ' + ' '.join(f'{b}:{bins[b][2] / bins[b][0]:.2f}(n={bins[b][0]})' for b in sorted(bins)))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'all'
    splits = ['dev', 'held', 'confirm']
    for a in sys.argv[2:]:
        if a.startswith('--splits='):
            splits = a.split('=')[1].split(',')
    tables, U = load()
    t0 = time.time()
    if cmd in ('corpus', 'all'):
        cmd_corpus(tables, U)
    if cmd in ('extract', 'all'):
        cmd_extract(tables, U, splits)
        cmd_abstain(tables, U, splits)
    if cmd in ('verify', 'all'):
        import k3_main_verify as V
        V.run(tables, U, splits, log)
    if cmd in ('controls', 'all'):
        import k3_main_controls as C
        C.run(tables, U, splits, log)
    if cmd in ('struct', 'all'):
        import k3_struct as S
        S.run(log)
        import k3_realtext as R
        R.run(log)
    if cmd in ('down', 'all'):
        import k3_main_down as D
        D.run(tables, U, splits, log)
    log('elapsed', round(time.time() - t0, 1))
    open(os.path.join(OUT, f'k3_{cmd}.log'), 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
