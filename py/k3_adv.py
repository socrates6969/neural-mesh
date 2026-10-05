"""k3_adv: adversarial / realistic-failure layouts built from REAL table grids. Each returns (text, gold_facts_override or None) so scoring can use the original gold
except where the perturbation changes truth (blank cells remove gold cells; transposed swaps key roles)."""
import re, random, textwrap
from k3_data import strip_md, fnv, norm_key, core_key
from k3_render import apply_vfmt, ws_block, sanitize, grid_cols_numeric

ADV = ['transposed', 'blank_cells', 'footnote_marks', 'glued_units', 'ocr_1pct', 'side_by_side', 'wrapped_labels', 'repeated_header']
ADV_ABLATE = ['no_distractors']


def _ctx(T):
    cb = [l for l in T['ctx_before'] if '|' not in l and l.strip()]
    ca = [l for l in T['ctx_after'] if '|' not in l and l.strip()]
    return cb, ca


def render_adv(T, kind, seed, partner=None):
    rng = random.Random(fnv(T['id'] + kind) ^ seed)
    hdr, rows, _ = apply_vfmt(T, 'v0', rng)
    ncol = len(hdr)
    numcol = grid_cols_numeric(T, hdr, rows)
    gold = dict(T['facts'])
    cb, ca = _ctx(T)
    lines = None
    if kind == 'transposed':
        # header row = row labels; first column = original headers. Needs a small table
        if len(rows) > 12 or ncol < 3:
            return None
        newhdr = [hdr[0]] + [r[0] for r in rows]
        newrows = [[hdr[ci]] + [rows[ri][ci] for ri in range(len(rows))] for ci in range(1, ncol)]
        nn = [True] * len(newhdr)
        nn[0] = False
        lines = ws_block([[sanitize(h) for h in newhdr]], newrows, nn)
        # truth: original (row, col) now appears as (col-as-row, row-as-col). Extractor reading the text as-is will emit swapped keys; truth is the swapped gold.
        gold = {(ck, rk): v for (rk, ck), v in T['facts'].items()}
    elif kind == 'blank_cells':
        rows = [list(r) for r in rows]
        n = 0
        for ri in range(len(rows)):
            for ci in range(1, ncol):
                if (T['rk'][ri], T['hk'][ci]) in gold and rng.random() < 0.12:
                    rows[ri][ci] = ''
                    del gold[(T['rk'][ri], T['hk'][ci])]
                    n += 1
        lines = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
    elif kind == 'footnote_marks':
        rows = [list(r) for r in rows]
        for ri in range(len(rows)):
            for ci in range(1, ncol):
                if (T['rk'][ri], T['hk'][ci]) in gold and rng.random() < 0.15:
                    rows[ri][ci] = rows[ri][ci] + rng.choice(['*', '†', '[1]', '¹', ' (a)'])
        lines = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
    elif kind == 'glued_units':
        rows = [list(r) for r in rows]
        for ri in range(len(rows)):
            for ci in range(1, ncol):
                rows[ri][ci] = re.sub(r'^(\S+) ([A-Za-zµμ/%]+)$', r'\1\2', rows[ri][ci].strip())
        lines = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
    elif kind == 'ocr_1pct':
        swap = {'0': 'O', '1': 'l', '5': 'S', '8': 'B', '6': '5', '3': '8', '9': '4', '7': '1'}
        rows = [list(r) for r in rows]
        for ri in range(len(rows)):
            for ci in range(1, ncol):
                if (T['rk'][ri], T['hk'][ci]) in gold:
                    cell = rows[ri][ci]
                    out = []
                    for ch in cell:
                        if ch in swap and rng.random() < 0.01:
                            out.append(swap[ch])
                        else:
                            out.append(ch)
                    rows[ri][ci] = ''.join(out)
        lines = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
    elif kind == 'side_by_side':
        if partner is None:
            return None
        h2, r2, _ = apply_vfmt(partner, 'v0', rng)
        n2 = grid_cols_numeric(partner, h2, r2)
        la = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
        lb = ws_block([[sanitize(h) for h in h2]], r2, n2)
        wa = max(len(x) for x in la)
        m = max(len(la), len(lb))
        lines = [(la[i] if i < len(la) else '').ljust(wa) + '    ' + (lb[i] if i < len(lb) else '') for i in range(m)]
        lines = [l.rstrip() for l in lines]
        g2 = dict(partner['facts'])
        # keys of the two tables may collide: only keep gold keys unique to either side
        for k, v in g2.items():
            if k in gold:
                del gold[k]
            else:
                gold[k] = v
    elif kind == 'wrapped_labels':
        lab_w = 14
        fr_all = [textwrap.wrap(sanitize(r[0]), width=lab_w, break_long_words=False) or [''] for r in rows]
        rows2 = [[fr_all[ri][0]] + list(rows[ri][1:]) for ri in range(len(rows))]
        base = ws_block([[sanitize(h) for h in hdr]], rows2, numcol)
        lines = [base[0]]
        for ri in range(len(rows)):
            lines.append(base[1 + ri])
            for fr in fr_all[ri][1:]:
                lines.append(fr)
    elif kind == 'repeated_header':
        base = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
        hl = base[0]
        body = base[1:]
        lines = [hl] + body[:max(3, len(body) // 2)] + ['', 'Table (continued)', hl] + body[max(3, len(body) // 2):]
    else:
        raise ValueError(kind)
    text = '\n'.join(cb + [''] + lines + [''] + ca) + '\n'
    return {'text': text, 'gold': gold}
