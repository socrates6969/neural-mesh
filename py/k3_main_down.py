"""downstream: lookup / metric questions answered (a) by a plain-reader NOISE MODEL (assumption constants below, swept) vs (b) from extracted values, with / without verification.
Questions & truths come from the REAL grid; renderings are synthetic; reader noise is an ASSUMPTION (h19 constants reused, labelled)."""
import collections, json, math, os, random, re
from k3_main_helpers import *
from k3_data import fnv

# ---- reader-noise model constants (ASSUMPTIONS, copied from h19 down.rs and swept) -------
E0, A_DIS, B_LEN, C_SCALE, C_LAY, E_ARITH = 0.010, 0.0004, 0.006, 0.030, 0.006, 0.020
# ---- costs (units of one full-document read) -------
C_TOOL, C_SPEC, C_VER = 0.05, 0.10, 0.02
E_SPEC = 0.02
LAYOUT_DIFF = {'pipe': 0, 'ws': 1, 'csv': 1, 'wrap': 2, 'split': 2, 'merged': 3, 'F1': 1, 'F2': 1, 'F3': 1, 'H1': 1, 'H2': 1, 'H3': 1, 'H4': 1}
# deployment mixture (ASSUMPTION): how documents arrive
MIX = {'pipe': 0.25, 'ws': 0.15, 'csv': 0.10, 'wrap': 0.10, 'F2': 0.05, 'H1': 0.10, 'H2': 0.10, 'H3': 0.05, 'split': 0.05, 'merged': 0.05}
LAYS = list(MIX)
VF = ['v0', 'mix']


def make_questions(T, rng):
    g = T['facts']
    cols = collections.defaultdict(list)
    for (rk, ck), f in g.items():
        cols[ck].append(rk)
    qs = []
    keys = sorted(g)
    for k in rng.sample(keys, min(2, len(keys))):
        qs.append({'type': 'lookup', 'needs': [k]})
    for ck, rks in sorted(cols.items()):
        if len(rks) >= 3:
            vals = [g[(rk, ck)]['value'] for rk in rks]
            top = max(vals)
            if vals.count(top) == 1:
                qs.append({'type': 'argmax', 'needs': [(rk, ck) for rk in rks]})
            qs.append({'type': 'col_sum', 'needs': [(rk, ck) for rk in rks]})
            qs.append({'type': 'col_mean', 'needs': [(rk, ck) for rk in rks]})
            break
    cl = sorted(cols)
    if len(cl) >= 2:
        rows = [rk for rk in sorted({r for (r, c) in g}) if (rk, cl[0]) in g and (rk, cl[1]) in g and g[(rk, cl[0])]['value'] != 0]
        if rows:
            rk = rng.choice(rows)
            qs.append({'type': 'ratio', 'needs': [(rk, cl[0]), (rk, cl[1])]})
            qs.append({'type': 'growth', 'needs': [(rk, cl[0]), (rk, cl[1])]})
    return qs


def compute(q, val):
    """val: key -> float. returns answer or None"""
    t = q['type']
    try:
        if t == 'lookup':
            return val(q['needs'][0])
        if t == 'argmax':
            vs = [(val(k), k[0]) for k in q['needs']]
            return max(vs, key=lambda x: x[0])[1]
        if t == 'col_sum':
            return sum(val(k) for k in q['needs'])
        if t == 'col_mean':
            return sum(val(k) for k in q['needs']) / len(q['needs'])
        if t == 'ratio':
            return val(q['needs'][0]) / val(q['needs'][1])
        if t == 'growth':
            a, b = val(q['needs'][0]), val(q['needs'][1])
            return (b - a) / a
    except (ZeroDivisionError, KeyError, TypeError):
        return None
    return None


def same(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return close(a, b, 1e-6)


def nops(q):
    t = q['type']
    return {'lookup': 0, 'argmax': len(q['needs']) - 1, 'col_sum': len(q['needs']) - 1, 'col_mean': len(q['needs']), 'ratio': 1, 'growth': 2}[t]


def eval_render(T, lay, vf, know, rng_seed, xs_cache=None):
    r = render_table(T, lay, vf, SEED)
    if r is None:
        return None
    text = r['text']
    facts = X.extract(text, know)
    best = best_by_key(facts)
    ver = {n: verify_facts(text, list(best.values()), vc) for n, vc in (('full', VFULL), ('narrow', VNARROW))}
    # second, independent clean rendering (csv/v0, other seed) for the cross-source arm  [ASSUMPTION: a second clean copy exists]
    r2 = render_table(T, 'csv', 'v0', SEED + 1)
    best2 = best_by_key(X.extract(r2['text'], X.KNOW_V3)) if r2 else {}
    g = T['facts']
    rng = random.Random(fnv(T['id']) ^ 77)
    qs = make_questions(T, rng)
    nnum_text = len(re.findall(r'\d+(?:[.,]\d+)?', text))
    out = []
    for q in qs:
        truth = compute(q, lambda k: g[k]['value'])
        if truth is None:
            continue
        needs = q['needs']
        # plain-reader components
        scale_n = 0
        for k in needs:
            cls = gclass(g[k])
            ci = g[k]['ci']
            changed = bool(r['chosen'].get(ci))
            scale_n += (cls != 'plain') or changed
        comp = {'nreads': len(needs), 'scale': scale_n, 'ops': nops(q), 'lay': LAYOUT_DIFF[lay], 'ndist': min(150, max(0, nnum_text - len(needs))), 'nchars': len(text)}
        arms = {}
        have = all(k in best for k in needs)
        if have:
            ans = compute(q, lambda k: best[k].v)
            right = same(ans, truth)
            low = any(best[k].conf < 0.5 for k in needs)
            fl = {n: any(is_flagged(ver[n][id(best[k])]['flags']) for k in needs) for n in ('full', 'narrow')}
            # repaired answer
            def repv(k, n='full'):
                rp = ver[n][id(best[k])]['repair']
                return rp if (is_flagged(ver[n][id(best[k])]['flags']) and rp is not None) else best[k].v
            ans_r = compute(q, lambda k: repv(k))
            # cross-source: any needed key conflicting with the 2nd rendering
            conflict = any((k in best2) and not close(best[k].v, best2[k].v, 1e-6) for k in needs)
            single = sum(1 for k in needs if k not in best2)
            arms = {'ext': ('ans', right), 'tau': ('fb', None) if low else ('ans', right),
                    'ver_full': ('fb', None) if fl['full'] else ('ans', right),
                    'ver_full_rep': ('fb', None) if (fl['full'] and any(ver['full'][id(best[k])]['repair'] is None and is_flagged(ver['full'][id(best[k])]['flags']) for k in needs)) else ('ans', same(ans_r, truth)),
                    'ver_narrow': ('fb', None) if fl['narrow'] else ('ans', right),
                    'ver_full_xsrc': ('fb', None) if (fl['full'] or conflict) else ('ans', right),
                    'ver_rand15': ('fb', None) if (rng.random() < 1 - (1 - 0.15) ** len(needs)) else ('ans', right)}
        else:
            arms = {a: ('fb', None) for a in ('ext', 'tau', 'ver_full', 'ver_full_rep', 'ver_narrow', 'ver_full_xsrc', 'ver_rand15')}
        out.append({'qtype': q['type'], 'comp': comp, 'arms': arms})
    return out


def plain_p(comp, noise):
    e = noise * (E0 + A_DIS * comp['ndist'] / max(1, comp['nreads']) * 0 + C_LAY * comp['lay'] + B_LEN * math.log(1 + comp['nchars'] / 800) / max(1, comp['nreads']) * 0)
    # per-fact misread
    pf = noise * (E0 + C_LAY * comp['lay'])
    p = (1 - min(0.5, pf)) ** comp['nreads']
    p *= (1 - min(0.5, noise * C_SCALE)) ** comp['scale']
    p *= (1 - min(0.5, noise * B_LEN * math.log(1 + comp['nchars'] / 800)))
    p *= (1 - min(0.5, noise * A_DIS * comp['ndist']))
    p *= (1 - min(0.5, noise * E_ARITH)) ** comp['ops']
    return p


def arm_value(arm, status, comp, noise, spec_c=C_SPEC, tool_c=C_TOOL, spec_err=E_SPEC):
    """-> (expected accuracy, cost)"""
    if arm == 'plain':
        return plain_p(comp, noise), 1.0
    kind, right = status
    c = tool_c + spec_c + (C_VER if arm.startswith('ver') else 0.0) + (tool_c if arm == 'ver_full_xsrc' else 0.0)
    pp = plain_p(comp, noise)
    if kind == 'ans':
        return (1 - spec_err * noise) * (1.0 if right else 0.0), c
    return (1 - spec_err * noise) * pp + spec_err * noise * 0.0, c + 1.0


ARMS = ['plain', 'ext', 'tau', 'ver_full', 'ver_full_rep', 'ver_narrow', 'ver_full_xsrc', 'ver_rand15']


def aggregate(rows, noise, mixw=None, spec_c=C_SPEC, spec_err=E_SPEC):
    """rows: list of (doc, lay, qrec). returns per-arm dict doc -> [sum_acc, n] and cost sums"""
    acc = {a: collections.defaultdict(lambda: [0.0, 0.0]) for a in ARMS}
    cost = {a: [0.0, 0.0] for a in ARMS}
    stat = {a: collections.Counter() for a in ARMS}
    for doc, lay, q in rows:
        w = 1.0 if mixw is None else mixw.get(lay, 0.0)
        if w == 0:
            continue
        for a in ARMS:
            st = q['arms'].get(a) if a != 'plain' else None
            v, c = arm_value(a, st, q['comp'], noise, spec_c=spec_c, spec_err=spec_err)
            acc[a][doc][0] += w * v
            acc[a][doc][1] += w
            cost[a][0] += w * c
            cost[a][1] += w
            if a != 'plain':
                stat[a]['ans_right' if (st[0] == 'ans' and st[1]) else ('ans_wrong' if st[0] == 'ans' else 'fb')] += w
                stat[a]['n'] += w
    return acc, cost, stat


def run(tables, U, splits, log):
    out = {}
    GP = []
    log('\n## D DOWNSTREAM QA (questions from the REAL grid: lookup, argmax, column sum/mean, ratio, growth; reader = ASSUMED noise model, extraction arms exact)')
    log('   noise constants (ASSUMPTIONS, h19 values): E0=%.3f A_DIS=%.4f B_LEN=%.3f C_SCALE=%.3f C_LAY=%.3f E_ARITH=%.3f ; costs tool %.2f spec %.2f verifier %.2f fallback full read 1.0 ; spec failure E_SPEC=%.2f' % (E0, A_DIS, B_LEN, C_SCALE, C_LAY, E_ARITH, C_TOOL, C_SPEC, C_VER, E_SPEC))
    for kn in ('v1', 'v3'):
        for sp in splits:
            sel = [T for T in U if T['split'] == sp]
            rows = []
            for T in sel:
                for lay in LAYS:
                    for vf in VF:
                        res = eval_render(T, lay, vf, KNOWS[kn], 0)
                        if res is None:
                            continue
                        for q in res:
                            q['vf'] = vf
                            rows.append((T['doc'], lay, q))
            out[f'{kn}_{sp}_nq'] = len(rows)
            for label, mw in (('MIX(deployment mixture, 50% clean/50% perturbed values)', MIX), ('pipe-only (native markdown)', {'pipe': 1.0}), ('HARD(ws,wrap,csv,H1-3,split,merged)', {'ws': 1, 'wrap': 1, 'csv': 1, 'H1': 1, 'H2': 1, 'H3': 1, 'split': 1, 'merged': 1})):
                for noise in (1.0,):
                    acc, cost, stat = aggregate(rows, noise, mw)
                    base = {d: tuple(v) for d, v in acc['plain'].items()}
                    log(f'  extractor {kn} split {sp} cond {label} noise x{noise} (questions {len(rows)})')
                    pvals = []
                    names = []
                    for a in ARMS:
                        da = {d: tuple(v) for d, v in acc[a].items()}
                        tot = sum(v[0] for v in da.values()) / max(1e-9, sum(v[1] for v in da.values()))
                        cst = cost[a][0] / max(1e-9, cost[a][1])
                        if a == 'plain':
                            log(f'     {a:14s} acc {tot:.4f} cost {cst:.3f}')
                            continue
                        d, lo, hi, p = paired_boot(da, base)
                        pvals.append(p)
                        GP.append((f'{kn}/{sp}/{label[:4]}/{a}', d, p))
                        names.append(a)
                        s = stat[a]
                        n = s['n']
                        log(f'     {a:14s} acc {tot:.4f} diff-vs-plain {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f} cost {cst:.3f} | answered-right {s["ans_right"] / n:.3f} answered-WRONG(silent) {s["ans_wrong"] / n:.3f} fallback {s["fb"] / n:.3f}')
                        out[f'{kn}_{sp}_{label[:4]}_{a}'] = {'acc': tot, 'diff': d, 'ci': [lo, hi], 'p': p, 'cost': cst, 'silent_wrong': s['ans_wrong'] / n, 'fallback': s['fb'] / n}
                    adj = holm(pvals)
                    log('       Holm-adjusted p (over arms, this cond): ' + ', '.join(f'{n_}:{a_:.4f}' for n_, a_ in zip(names, adj)))
            # per-layout table
            log(f'  extractor {kn} split {sp}: per-layout accuracy (noise x1, vf=all) plain / ext / ext+ver_full / ext+ver_full_rep / ver_narrow / cost(ver_full)')
            for lay in LAYS:
                acc, cost, stat = aggregate(rows, 1.0, {lay: 1.0})
                line = []
                for a in ('plain', 'ext', 'ver_full', 'ver_full_rep', 'ver_narrow'):
                    tot = sum(v[0] for v in acc[a].values()) / max(1e-9, sum(v[1] for v in acc[a].values()))
                    line.append(f'{tot:.3f}')
                log(f'     {lay:6s} ' + ' / '.join(line) + f'  cost {cost["ver_full"][0] / max(1e-9, cost["ver_full"][1]):.2f}')
            # per question type
            log(f'  extractor {kn} split {sp}: per-question-type accuracy under MIX (plain / ext / ver_full)')
            for qt in ('lookup', 'argmax', 'col_sum', 'col_mean', 'ratio', 'growth'):
                sub = [(d, l, q) for d, l, q in rows if q['qtype'] == qt]
                acc, cost, stat = aggregate(sub, 1.0, MIX)
                line = []
                for a in ('plain', 'ext', 'ver_full'):
                    tot = sum(v[0] for v in acc[a].values()) / max(1e-9, sum(v[1] for v in acc[a].values()))
                    line.append(f'{tot:.3f}')
                log(f'     {qt:9s} n={len(sub):5d} ' + ' / '.join(line))
            # sensitivity to reader noise and costs
            log(f'  extractor {kn} split {sp}: SENSITIVITY under MIX: reader-noise multiplier -> plain / ext / ver_full / ver_full_rep (cost ver_full)')
            for noise in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                acc, cost, stat = aggregate(rows, noise, MIX)
                line = []
                for a in ('plain', 'ext', 'ver_full', 'ver_full_rep'):
                    tot = sum(v[0] for v in acc[a].values()) / max(1e-9, sum(v[1] for v in acc[a].values()))
                    line.append(f'{tot:.4f}')
                log(f'     x{noise:<5} ' + ' / '.join(line) + f'  cost {cost["ver_full"][0] / max(1e-9, cost["ver_full"][1]):.2f}')
                out[f'{kn}_{sp}_sens_{noise}'] = line
            for spec_c in (0.05, 0.10, 0.30):
                acc, cost, stat = aggregate(rows, 1.0, MIX, spec_c=spec_c)
                log(f'     spec-step cost {spec_c}: cost ver_full {cost["ver_full"][0] / max(1e-9, cost["ver_full"][1]):.3f} (plain 1.000)')
            for spec_err in (0.0, 0.02, 0.10):
                acc, cost, stat = aggregate(rows, 1.0, MIX, spec_err=spec_err)
                line = [sum(v[0] for v in acc[a].values()) / max(1e-9, sum(v[1] for v in acc[a].values())) for a in ('plain', 'ext', 'ver_full')]
                log(f'     spec-step error E_SPEC={spec_err}: plain/ext/ver_full {line[0]:.4f}/{line[1]:.4f}/{line[2]:.4f}')
    adj = holm([g[2] for g in GP])
    sig = [(g[0], round(g[1], 4), a) for g, a in zip(GP, adj) if a < 0.05]
    ns = [(g[0], round(g[1], 4), round(g[2], 4), round(a, 4)) for g, a in zip(GP, adj) if a >= 0.05]
    log(f'  GLOBAL Holm over all {len(GP)} (extractor x split x condition x arm) paired-bootstrap tests vs plain (normal-approx p from bootstrap SE): {len(sig)} significant at 0.05; NOT significant: {len(ns)}')
    for x in ns:
        log(f'     not-sig {x}')
    json.dump(out, open(os.path.join(OUT, 'k3_down.json'), 'w'), indent=1, default=str)
