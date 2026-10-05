"""verifier evaluation: catch vs false-flag, narrow vs full knowledge, reasons, injected errors (scale/sign/transposition/decimal/wrong-cell), footing applicability, cross-source."""
import collections, random, math, re, json, os
from k3_main_helpers import *


def pct(a, b):
    return a / b if b else float('nan')


def ablation(U, splits, log, out):
    log('\n## V0 verifier check ablation (extractor v1, layouts pipe/ws/wrap/csv/F2, value formats v0+mix+eu): catch of wrong / false-flag on correct, per check set. DEV is where the default (all hard checks, outlier OFF) was frozen.')
    variants = {'all_hard': (VCfg(), HARD_FLAGS), 'no_span': (VCfg(span=False), HARD_FLAGS), 'no_amb': (VCfg(amb=False), HARD_FLAGS - {'ambiguous_sep'}),
                'no_footing': (VCfg(footing=False), HARD_FLAGS), '+magnitude_outlier': (VCfg(outlier=True), HARD_FLAGS | {'magnitude_outlier'}), 'narrow_units': (VNARROW, HARD_FLAGS)}
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        rs = run_grid(sel, ['pipe', 'ws', 'wrap', 'csv', 'F2'], ['v0', 'mix', 'eu'], X.KNOW_V1, vcfgs={k: v[0] for k, v in variants.items()})
        row = []
        for k, (vc, fs) in variants.items():
            c = collections.Counter()
            for rd in rs:
                for rc in rd['recs']:
                    if rc['kind'] == 'fn':
                        continue
                    fl = is_flagged(rc['flags'][k], fs)
                    if rc['kind'] == 'tp':
                        c['ok'] += 1
                        c['okf'] += fl
                    else:
                        c['bad'] += 1
                        c['badf'] += fl
            row.append(f'{k}: catch {pct(c["badf"], c["bad"]):.3f} ff {pct(c["okf"], c["ok"]):.4f}')
        log(f'  {sp:7s} ' + ' | '.join(row))
        out[f'ablation_{sp}'] = row


def run(tables, U, splits, log):
    out = {}
    ablation(U, splits, log, out)
    log('\n## V1 verifier catch / false-flag on NATURAL extraction errors (extractor v1 = dev knowledge only; layouts pipe,ws,wrap,csv,F2; value formats v0+mix+eu)')
    log('   catch = flagged / wrong-extracted ; false-flag = flagged / correct-extracted ; repair = flagged wrong facts for which the verifier proposes the TRUE value')
    for kn in ('v1', 'v3'):
        for sp in splits:
            sel = [T for T in U if T['split'] == sp]
            rs = run_grid(sel, ['pipe', 'ws', 'wrap', 'csv', 'F2'], ['v0', 'mix', 'eu'], KNOWS[kn])
            for vfl, vfs in (('v0+mix', ('v0', 'mix')), ('eu', ('eu',))):
                rsx = [rd for rd in rs if rd['vfmt'] in vfs]
                for vn in ('full', 'narrow'):
                    # doc-level bootstrap for catch/false-flag
                    d_wrong = collections.defaultdict(lambda: [0, 0, 0])
                    d_ok = collections.defaultdict(lambda: [0, 0])
                    reasons_w, reasons_ok = collections.Counter(), collections.Counter()
                    junk = [0, 0]
                    for rd in rsx:
                        for rc in rd['recs']:
                            if rc['kind'] == 'fn':
                                continue
                            fl = is_flagged(rc['flags'][vn])
                            if rc['kind'] == 'tp':
                                d_ok[rd['doc']][0] += 1
                                d_ok[rd['doc']][1] += fl
                                if fl:
                                    for r in rc['flags'][vn] & HARD_FLAGS:
                                        reasons_ok[r] += 1
                            elif rc['kind'] == 'wrong':
                                d_wrong[rd['doc']][0] += 1
                                d_wrong[rd['doc']][1] += fl
                                d_wrong[rd['doc']][2] += (fl and rc['rep_ok'][vn])
                                if fl:
                                    for r in rc['flags'][vn] & HARD_FLAGS:
                                        reasons_w[r] += 1
                            else:
                                junk[0] += 1
                                junk[1] += fl
                    cw, lw, hw = boot_ratio(d_wrong, lambda s: s[1] / s[0] if s[0] else float('nan'), B=300)
                    co, lo, ho = boot_ratio(d_ok, lambda s: s[1] / s[0] if s[0] else float('nan'), B=300)
                    nw = sum(v[0] for v in d_wrong.values())
                    nr = sum(v[2] for v in d_wrong.values())
                    nok = sum(v[0] for v in d_ok.values())
                    log(f'  {kn} {sp:7s} {vfl:7s} verifier={vn:6s}: wrong={nw:6d} CATCH {cw:.3f} [{lw:.3f},{hw:.3f}] repaired-to-truth {pct(nr, nw):.3f} | correct={nok:6d} FALSE-FLAG {co:.4f} [{lo:.4f},{ho:.4f}] | junk(non-gold key) {junk[1]}/{junk[0]}')
                    log(f'       reasons on wrong: {dict(reasons_w)}  reasons on correct: {dict(reasons_ok)}')
                    out[f'{kn}_{sp}_{vfl}_{vn}'] = {'wrong': nw, 'catch': cw, 'catch_ci': [lw, hw], 'ff': co, 'ff_ci': [lo, ho], 'repaired': pct(nr, nw), 'ok': nok}
                # P/R before and after verification (full), drop-flagged vs drop+repair
                base = summarize(doc_counts(rsx, lambda r: 'ALL'), B=100)['ALL']
                a1 = summarize(doc_counts(rsx, lambda r: 'ALL', vname='full'), B=100)['ALL']
                a2 = summarize(doc_counts(rsx, lambda r: 'ALL', vname='full', repair=True), B=100)['ALL']
                a3 = summarize(doc_counts(rsx, lambda r: 'ALL', vname='narrow'), B=100)['ALL']
                log(f'       {kn} {sp} {vfl}: P/R/F1 raw {base["P"]:.3f}/{base["R"]:.3f}/{base["f1"]:.3f} | full-ver drop-flagged {a1["P"]:.3f}/{a1["R"]:.3f}/{a1["f1"]:.3f} | full-ver +repair {a2["P"]:.3f}/{a2["R"]:.3f}/{a2["f1"]:.3f} | narrow drop {a3["P"]:.3f}/{a3["R"]:.3f}/{a3["f1"]:.3f}')
                out[f'{kn}_{sp}_{vfl}_pr'] = {'raw': base, 'full_drop': a1, 'full_repair': a2, 'narrow_drop': a3}
    log('\n## V2 INJECTED errors into clean extractions (v3 on pipe/v0 renders; facts extracted correctly, then ONE field corrupted): catch by verifier (full / narrow)')
    rng = random.Random(5)
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        cnt = collections.defaultdict(lambda: [0, 0, 0])  # type -> [n, full_flag, narrow_flag]
        ff = [0, 0, 0]
        for T in sel:
            r = render_table(T, 'pipe', 'v0', SEED)
            facts = list(best_by_key(X.extract(r['text'], X.KNOW_V3)).values())
            facts = [f for f in facts if f.key() in T['facts'] and close(f.v, T['facts'][f.key()]['value'], 1e-6)]
            if len(facts) < 2:
                continue
            base_f = {n: verify_facts(r['text'], facts, vc) for n, vc in (('full', VFULL), ('narrow', VNARROW))}
            ff[0] += len(facts)
            ff[1] += sum(is_flagged(base_f['full'][id(f)]['flags']) for f in facts)
            ff[2] += sum(is_flagged(base_f['narrow'][id(f)]['flags']) for f in facts)
            idx = list(range(len(facts)))
            rng.shuffle(idx)
            for i in idx[:max(3, len(facts) // 5)]:
                f = facts[i]
                for typ in ('scale_x1000', 'scale_div1000', 'sign_flip', 'digit_transpose', 'decimal_div10'):
                    v0 = f.v
                    if typ == 'scale_x1000':
                        nv = v0 * 1000
                    elif typ == 'scale_div1000':
                        nv = v0 / 1000
                    elif typ == 'sign_flip':
                        if v0 == 0:
                            continue
                        nv = -v0
                    elif typ == 'digit_transpose':
                        s = re.sub(r'\D', '', repr(abs(v0)))
                        if len(s) < 2 or s[0] == s[1]:
                            continue
                        s2 = s[1] + s[0] + s[2:]
                        nv = float(s2) / float(s) * v0 if float(s) else v0
                        if close(nv, v0):
                            continue
                    else:
                        nv = v0 / 10
                    if v0 == 0:
                        continue
                    f.v = nv
                    for n, vc in (('full', VFULL), ('narrow', VNARROW)):
                        res = verify_facts(r['text'], facts, vc)
                        cnt[typ][0 if n == 'full' else 0] += 0
                        cnt[typ][1 if n == 'full' else 2] += is_flagged(res[id(f)]['flags'])
                    cnt[typ][0] += 1
                    f.v = v0
        log(f'  {sp:7s} baseline false-flag on clean facts: full {pct(ff[1], ff[0]):.4f} narrow {pct(ff[2], ff[0]):.4f} (n={ff[0]})')
        for typ, v in cnt.items():
            log(f'     inject {typ:16s} n={v[0]:5d} catch full {pct(v[1], v[0]):.3f} narrow {pct(v[2], v[0]):.3f}   [value mutated, span text left unchanged]')
        out[f'inject_{sp}'] = {k: v for k, v in cnt.items()}

    log('\n## V3 WRONG-CELL errors (value AND span move to a neighbouring cell in the same row; the hard class: span re-parse agrees with itself): catch by footing/other checks')
    rng = random.Random(9)
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        n = [0, 0, 0]
        foot_tab = 0
        for T in sel:
            r = render_table(T, 'pipe', 'v0', SEED)
            facts = list(best_by_key(X.extract(r['text'], X.KNOW_V3)).values())
            facts = [f for f in facts if f.key() in T['facts'] and close(f.v, T['facts'][f.key()]['value'], 1e-6)]
            byrow = collections.defaultdict(list)
            for f in facts:
                byrow[f.rk].append(f)
            foot_tab += footing_applicable(facts) > 0
            for rk, fs in byrow.items():
                if len(fs) < 2 or rng.random() > 0.3:
                    continue
                a, b = rng.sample(fs, 2)
                if close(a.v, b.v):
                    continue
                av, asp, braw = a.v, a.span, a.raw
                a.v, a.span, a.raw = b.v, b.span, b.raw
                for nm, vc in (('full', VFULL), ('narrow', VNARROW)):
                    res = verify_facts(r['text'], facts, vc)
                    if nm == 'full':
                        n[0] += 1
                    n[1 if nm == 'full' else 2] += is_flagged(res[id(a)]['flags'])
                a.v, a.span, a.raw = av, asp, braw
        log(f'  {sp:7s} wrong-cell n={n[0]} catch full {pct(n[1], n[0]):.3f} narrow {pct(n[2], n[0]):.3f}; tables with a total row/col (footing applicable): {foot_tab}/{len(sel)}')
        out[f'wrongcell_{sp}'] = n

    log('\n## V4 footing check on REAL tables with total rows/cols (strict 1% tolerance vs precision-aware tolerance): how often does a real "total" foot?')
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        res = {'applicable': 0, 'strict_fail': 0, 'precise_fail': 0}
        for T in sel:
            r = render_table(T, 'pipe', 'v0', SEED)
            facts = list(best_by_key(X.extract(r['text'], X.KNOW_V3)).values())
            if footing_applicable(facts) == 0:
                continue
            res['applicable'] += 1
            for nm, vc in (('strict', VCfg(footing_precise=False)), ('precise', VCfg())):
                v = verify_facts(r['text'], facts, vc)
                if any('footing' in x['flags'] for x in v.values()):
                    res[nm + '_fail'] += 1
        log(f'  {sp:7s} {res}')
        out[f'footing_{sp}'] = res

    log('\n## V5 cross-source agreement between two renderings of the same table (v1 extractor, mix surface in A vs original surface in B vs identical surface)')
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        for name, (la, va, lb, vb) in {'same-surface ws/v0 vs F2/v0': ('ws', 'v0', 'F2', 'v0'), 'diff-surface ws/mix vs F2/v0': ('ws', 'mix', 'F2', 'v0'), 'diff-surface ws/mix vs csv/v0': ('ws', 'mix', 'csv', 'v0')}.items():
            c = collections.Counter()
            for T in sel:
                ra = render_table(T, la, va, SEED)
                rb = render_table(T, lb, vb, SEED + 1)
                if ra is None or rb is None:
                    continue
                fa = best_by_key(X.extract(ra['text'], X.KNOW_V1))
                fb = best_by_key(X.extract(rb['text'], X.KNOW_V1))
                agree, conflict = crosscheck(list(fa.values()), list(fb.values()))
                for k, f in fa.items():
                    g = T['facts'].get(k)
                    if g is None:
                        continue
                    ok = close(f.v, g['value'], 1e-6)
                    st = 'agree' if k in agree else ('conflict' if k in conflict else 'single')
                    c[(ok, st)] += 1
            wrong = sum(v for (ok, st), v in c.items() if not ok)
            corr = sum(v for (ok, st), v in c.items() if ok)
            log(f'  {sp:7s} {name:32s}: wrong A-facts {wrong}: caught by conflict {pct(c[(False, "conflict")], wrong):.3f}, agree-on-wrong {pct(c[(False, "agree")], wrong):.3f}, single-source {pct(c[(False, "single")], wrong):.3f} | correct {corr}: false conflict {pct(c[(True, "conflict")], corr):.4f}')
            out[f'xsrc_{sp}_{name}'] = {str(k): v for k, v in c.items()}
    json.dump(out, open(os.path.join(OUT, 'k3_verify.json'), 'w'), indent=1, default=str)
