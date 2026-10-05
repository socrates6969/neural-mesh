"""controls: (a) NO-STRUCTURE real prose (false-extraction rate), (b) adversarial / realistic-failure layouts built from real grids."""
import collections, glob, os, re, json
from k3_main_helpers import *
from k3_adv import render_adv, ADV
from k3_data import find_tables


def score_text(text, gold, know):
    facts = X.extract(text, know)
    best = best_by_key(facts)
    ver = {n: verify_facts(text, list(best.values()), vc) for n, vc in (('full', VFULL), ('narrow', VNARROW))}
    c = collections.Counter()
    for k, f in best.items():
        g = gold.get(k)
        fl = is_flagged(ver['full'][id(f)]['flags'])
        fln = is_flagged(ver['narrow'][id(f)]['flags'])
        if g is None:
            c['junk'] += 1
            c['junk_flag'] += fl
            c['junk_flag_narrow'] += fln
        elif close(f.v, g['value'], 1e-6):
            c['tp'] += 1
            c['tp_flag'] += fl
        else:
            c['wrong'] += 1
            c['wrong_flag'] += fl
            c['wrong_flag_narrow'] += fln
    c['fn'] = sum(1 for k in gold if k not in best or not close(best[k].v, gold[k]['value'], 1e-6))
    c['gold'] = len(gold)
    return c


def prose_lines(path):
    t = open(path, encoding='utf8', errors='ignore').read()
    tbs, lines = find_tables(t)
    skip = set()
    for tb in tbs:
        skip.update(range(tb['start'], tb['end']))
    return [l for i, l in enumerate(lines) if i not in skip]


def run(tables, U, splits, log):
    out = {}
    log('\n## C1 CONTROL: NO-STRUCTURE REAL PROSE (non-table lines of the same files; any extracted fact is a false extraction)')
    docs_by_split = collections.defaultdict(list)
    seen = set()
    for T in tables:
        if T['doc'] in seen:
            continue
        seen.add(T['doc'])
        path = os.path.join(REPO, T['doc'].lstrip('/')) if T['source'] == 'repo' else None
        if path is None:
            reg = glob.glob(os.path.expanduser('~/.cargo/registry/src/*/' + T['doc']))
            path = reg[0] if reg else None
        if path and os.path.exists(path):
            docs_by_split[T['split']].append(path)
    for kn in ('v1', 'v3'):
        for sp in splits:
            nd = nf = nnum = nlines = 0
            flagged = 0
            confs = []
            docs_hit = 0
            for p in docs_by_split[sp]:
                pl = prose_lines(p)
                # chunk into 15-line docs
                for i in range(0, len(pl), 15):
                    chunk = '\n'.join(pl[i:i + 15])
                    nums = len(re.findall(r'\d+(?:[.,]\d+)?', chunk))
                    if nums < 3:
                        continue
                    nd += 1
                    nnum += nums
                    facts = list(best_by_key(X.extract(chunk, KNOWS[kn])).values())
                    ver = verify_facts(chunk, facts, VFULL)
                    nf += len(facts)
                    docs_hit += len(facts) > 0
                    flagged += sum(is_flagged(ver[id(f)]['flags']) for f in facts)
                    confs += [f.conf for f in facts]
            log(f'  {kn} {sp:7s}: prose chunks with >=3 numbers {nd}, numbers {nnum}; chunks with >=1 false extraction {docs_hit} ({docs_hit / max(1, nd):.3f}); false facts {nf} ({1000 * nf / max(1, nnum):.1f} per 1000 numbers); flagged by verifier {flagged}/{nf}; mean conf {sum(confs) / max(1, len(confs)):.2f}; conf>=0.85: {sum(1 for c in confs if c >= 0.85)}')
            out[f'prose_{kn}_{sp}'] = {'chunks': nd, 'false_facts': nf, 'per1000': 1000 * nf / max(1, nnum), 'chunks_hit': docs_hit}

    log('\n## C2 ADVERSARIAL / realistic-failure layouts built from real grids (extractor v3; ws base layout; full verifier)')
    log('   silent-wrong = extracted facts with the right key-or-not but WRONG value / all extracted ; catch = flagged fraction of wrong+junk')
    for sp in splits:
        sel = [T for T in U if T['split'] == sp]
        for kind in ADV:
            tot = collections.Counter()
            nt = 0
            for i, T in enumerate(sel):
                partner = sel[(i + 1) % len(sel)] if kind == 'side_by_side' and len(sel) > 1 else None
                if partner is not None and partner['doc'] != T['doc']:
                    partner = next((q for q in sel if q['doc'] == T['doc'] and q is not T), None)
                r = render_adv(T, kind, SEED, partner)
                if r is None:
                    continue
                nt += 1
                tot.update(score_text(r['text'], r['gold'], X.KNOW_V3))
            p = tot['tp'] / max(1, tot['tp'] + tot['wrong'] + tot['junk'])
            rc = tot['tp'] / max(1, tot['gold'])
            ext = tot['tp'] + tot['wrong'] + tot['junk']
            bad = tot['wrong'] + tot['junk']
            bad_flag = tot['wrong_flag'] + tot['junk_flag']
            log(f'  {sp:7s} {kind:15s} tables={nt:3d} gold={tot["gold"]:5d} extracted={ext:5d}  P {p:.3f} R {rc:.3f}  silent-wrong+junk {bad} ({bad / max(1, ext):.3f} of extracted)  verifier catch {bad_flag / max(1, bad):.3f} (narrow {(tot["wrong_flag_narrow"] + tot["junk_flag_narrow"]) / max(1, bad):.3f})  false-flag on correct {tot["tp_flag"] / max(1, tot["tp"]):.3f}')
            out[f'adv_{sp}_{kind}'] = dict(tot)
    json.dump(out, open(os.path.join(OUT, 'k3_controls.json'), 'w'), indent=1, default=str)
