"""k3_render: render REAL markdown-table grids (k3_data) to layout-perturbed + value-format-perturbed text. Truth stays the original pipe grid.
Layouts: pipe ws wrap csv split(no-label continuation) merged(header merged, single-space) + prose families F1 F2 F3 (dev) / H1 H2 H3 H4 (held-out).
Value formats (all EXACT / value-preserving, applied only when representable): v0 thou sfx(K/M/B) neg(paren / unicode minus / trailing minus) pct unit eu mix."""
import re, csv, io, random, textwrap
from decimal import Decimal
from k3_data import strip_md, gold_parse_cell, fnv

LAYOUTS = ['pipe', 'ws', 'wrap', 'csv', 'split', 'merged', 'F1', 'F2', 'F3', 'H1', 'H2', 'H3', 'H4']
VFMTS = ['v0', 'thou', 'sfx', 'neg', 'pct', 'unit', 'eu', 'mix']
DEV_PROSE = ['F1', 'F2', 'F3']
HELD_PROSE = ['H1', 'H2', 'H3', 'H4']
LAYOUT_DIFFICULTY = {'pipe': 0, 'ws': 1, 'csv': 1, 'wrap': 2, 'split': 2, 'merged': 3, 'F1': 1, 'F2': 1, 'F3': 1, 'H1': 1, 'H2': 1, 'H3': 1, 'H4': 1}

PLAIN = re.compile(r'^[-+]?\d[\d,]*(\.\d+)?$|^[-+]?\.\d+$')
TIME_U = ['ns', 'µs', 'ms', 's']
TIME_P = {'ns': -9, 'µs': -6, 'μs': -6, 'us': -6, 'ms': -3, 's': 0}
SIZE_U = ['B', 'KB', 'MB', 'GB']
SIZE_P = {'B': 0, 'KB': 3, 'MB': 6, 'GB': 9}


def dstr(d):
    s = format(d.normalize(), 'f')
    return s


def shift(raw_num, p):
    """Decimal(raw_num) * 10**p as exact string"""
    return dstr(Decimal(raw_num.replace(',', '')) .scaleb(p))


def add_commas(s):
    neg = s.startswith('-')
    if neg:
        s = s[1:]
    ip, _, fp = s.partition('.')
    ip = re.sub(r'(?<=\d)(?=(\d{3})+$)', ',', ip)
    return ('-' if neg else '') + ip + ('.' + fp if fp else '')


def col_info(T, ci):
    cells = [(ri, T['rows'][ri][ci]) for ri in range(len(T['rows'])) if (T['rk'][ri], T['hk'][ci]) in T['facts']]
    return cells


def apply_vfmt(T, fmt, rng):
    """returns (header, rows) strings with value perturbations applied per column."""
    hdr = [strip_md(h) for h in T['header']]
    rows = [[strip_md(c) if j == 0 else c.strip() for j, c in enumerate(r)] for r in T['rows']]
    ncol = len(hdr)
    chosen = {}
    for ci in range(1, ncol):
        cells = col_info(T, ci)
        if not cells:
            continue
        raws = [strip_md(c) for _, c in cells]
        allplain = all(PLAIN.match(r) for r in raws)
        opts = []
        if allplain and any(abs(float(r.replace(',', ''))) >= 1000 for r in raws):
            opts += ['thou']
        if allplain and any(abs(float(r.replace(',', ''))) >= 1000 for r in raws):
            opts += ['sfx']
        if any(re.match(r'^-\d', r) for r in raws) and allplain:
            opts += ['neg']
        if all(re.fullmatch(r'\d+(\.\d+)?%', r) for r in raws):
            opts += ['pct']
        elif allplain and all(0 <= float(r.replace(',', '')) <= 1 and '.' in r and len(r.split('.')[1]) >= 2 for r in raws):
            opts += ['pct']
        um = [re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)\s*(ns|µs|μs|us|ms|s)', r) for r in raws]
        if all(um) and len(set(m.group(2) for m in um)) == 1:
            opts += ['unit']
        um = [re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)\s*(B|KB|MB|GB)', r) for r in raws]
        if all(um) and len(set(m.group(2) for m in um)) == 1:
            opts += ['unit']
        if allplain:
            opts += ['eu']
        if fmt == 'mix':
            sel = [o for o in opts if o != 'eu' and rng.random() < 0.6]
            chosen[ci] = sel[:2]
        elif fmt in opts:
            chosen[ci] = [fmt]
        else:
            chosen[ci] = []
    for ci, sel in chosen.items():
        cells = col_info(T, ci)
        for f in sel:
            if f == 'thou':
                for ri, c in cells:
                    r = strip_md(c)
                    rows[ri][ci] = add_commas(r.replace(',', ''))
            elif f == 'sfx':
                # per-column choice of one suffix so it stays consistent where exact
                for ri, c in cells:
                    r = strip_md(c).replace(',', '')
                    v = Decimal(r)
                    av = abs(v)
                    for suf, m in (('B', 9), ('M', 6), ('K', 3)):
                        if av >= Decimal(10) ** m:
                            d = v.scaleb(-m)
                            ds = dstr(d)
                            if len(ds.replace('-', '').replace('.', '')) <= 5:
                                rows[ri][ci] = ds + suf
                            break
            elif f == 'neg':
                style = rng.choice(['paren', 'umin', 'trail'])
                for ri, c in cells:
                    r = strip_md(c)
                    if r.startswith('-'):
                        body = r[1:]
                        rows[ri][ci] = {'paren': '(' + body + ')', 'umin': '−' + body, 'trail': body + '-'}[style]
            elif f == 'pct':
                for ri, c in cells:
                    r = strip_md(c)
                    if r.endswith('%'):
                        rows[ri][ci] = dstr(Decimal(r[:-1]).scaleb(-2))
                    else:
                        rows[ri][ci] = dstr(Decimal(r).scaleb(2)) + '%'
            elif f == 'unit':
                mode = rng.choice(['cell', 'hdr'])
                sample = strip_md(cells[0][1])
                ms = re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)\s*(ns|µs|μs|us|ms|s)', sample)
                if ms:
                    group, P = TIME_U, TIME_P
                    u0 = ms.group(2)
                else:
                    group, P = SIZE_U, SIZE_P
                    u0 = re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)\s*(B|KB|MB|GB)', sample).group(2)
                i0 = group.index('µs' if u0 in ('μs', 'us') else u0)
                u1 = group[rng.choice([j for j in (i0 - 1, i0 + 1) if 0 <= j < len(group)])]
                for ri, c in cells:
                    r = strip_md(c)
                    m = re.fullmatch(r'(\d[\d,]*(?:\.\d+)?)\s*([^\d\s]+)', r)
                    num, u = m.group(1), m.group(2)
                    new = shift(num, P[u] - P[u1])
                    rows[ri][ci] = new if mode == 'hdr' else new + ' ' + u1
                if mode == 'hdr':
                    hdr[ci] = hdr[ci] + ' (' + u1 + ')'
            elif f == 'eu':
                for ri, c in cells:
                    r = strip_md(c)
                    if ',' in r or '.' in r or abs(float(r.replace(',', ''))) >= 1000:
                        us = add_commas(r.replace(',', '')) if abs(float(r.replace(',', ''))) >= 1000 else r.replace(',', '')
                        rows[ri][ci] = us.replace(',', '\x00').replace('.', ',').replace('\x00', '.')
    return hdr, rows, chosen


def sanitize(c):
    return re.sub(r'\s+', ' ', c).strip()


def is_numish(c):
    return re.match(r'^[(\-−+~≈]?\$?\.?\d', sanitize(c)) is not None


def grid_cols_numeric(T, hdr, rows):
    n = len(hdr)
    out = [False] * n
    for ci in range(1, n):
        k = sum(1 for r in rows if is_numish(r[ci]))
        out[ci] = k >= max(1, len(rows) // 2)
    return out


def ws_block(hdr_lines, rows, numcol, widths=None, left_cols=None):
    """hdr_lines: list of lists (fragments per column, '' where none). Returns text lines."""
    n = len(numcol)
    cells_all = [[sanitize(c) for c in r] for r in rows]
    w = [0] * n
    for hl in hdr_lines:
        for j, f in enumerate(hl):
            w[j] = max(w[j], len(f))
    for r in cells_all:
        for j, c in enumerate(r):
            w[j] = max(w[j], len(c))

    def line(fr):
        parts = []
        for j in range(n):
            f = fr[j] if j < len(fr) else ''
            parts.append(f.rjust(w[j]) if numcol[j] else f.ljust(w[j]))
        return '  '.join(parts).rstrip()
    out = [line(hl) for hl in hdr_lines]
    out += [line(r) for r in cells_all]
    return out


def render_table(T, layout, vfmt, seed):
    rng = random.Random(fnv(T['id'] + layout + vfmt) ^ seed)
    hdr, rows, chosen = apply_vfmt(T, vfmt, random.Random(fnv(T['id'] + vfmt) ^ seed))  # same value perturbation across layouts
    ncol = len(hdr)
    numcol = grid_cols_numeric(T, hdr, rows)
    lines = []
    if layout == 'pipe':
        lines.append('| ' + ' | '.join(hdr) + ' |')
        lines.append('|' + '|'.join(['---'] * ncol) + '|')
        for r in rows:
            lines.append('| ' + ' | '.join(r) + ' |')
    elif layout == 'ws':
        lines = ws_block([[sanitize(h) for h in hdr]], rows, numcol)
    elif layout == 'wrap':
        wid = [max(6, min(11, max([len(sanitize(r[j])) for r in rows] + [4]))) for j in range(ncol)]
        frags = [textwrap.wrap(sanitize(hdr[j]), width=wid[j], break_long_words=False) or [''] for j in range(ncol)]
        nl = max(len(f) for f in frags)
        hl = [[(frags[j][i] if i < len(frags[j]) else '') for j in range(ncol)] for i in range(nl)]
        lines = ws_block(hl, rows, numcol)
    elif layout == 'csv':
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator='\n')
        w.writerow([sanitize(h) for h in hdr])
        for r in rows:
            w.writerow([sanitize(c) for c in r])
        lines = buf.getvalue().rstrip('\n').split('\n')
    elif layout == 'split':
        if ncol < 4:
            return None
        k = (ncol + 1) // 2
        a = ws_block([[sanitize(h) for h in hdr[:k]]], [r[:k] for r in rows], numcol[:k])
        b = ws_block([[sanitize(h) for h in hdr[k:]]], [r[k:] for r in rows], numcol[k:])
        lines = a + ['', '(continued; same row order)', ''] + b
    elif layout == 'merged':
        for r in rows:
            if any(' ' in sanitize(c) for c in r[1:]):
                return None
        if not all(numcol[1:]) or any((T['rk'][ri], T['hk'][ci]) not in T['facts'] for ri in range(len(rows)) for ci in range(1, ncol)):
            return None
        lines.append(' '.join(sanitize(h) for h in hdr))
        for r in rows:
            lines.append(' '.join(sanitize(c) for c in r))
    elif layout in DEV_PROSE + HELD_PROSE:
        facts = [(ri, ci) for ri in range(len(rows)) for ci in range(1, ncol) if (T['rk'][ri], T['hk'][ci]) in T['facts']]
        lab = lambda ri: sanitize(rows[ri][0])
        col = lambda ci: re.sub(r'\s*\([^)]*\)|\s*\[[^\]]*\]', '', sanitize(hdr[ci])).strip() or sanitize(hdr[ci])
        # keep unit-carrying header (moved unit) visible in the sentence
        colfull = lambda ci: sanitize(hdr[ci])
        by_row = {}
        for ri, ci in facts:
            by_row.setdefault(ri, []).append(ci)
        if layout == 'F1':
            for ri, cis in by_row.items():
                lines.append(lab(ri) + ': ' + '; '.join(f'{colfull(ci)} = {sanitize(rows[ri][ci])}' for ci in cis) + '.')
        elif layout == 'F2':
            lines.append(' '.join(f'For {lab(ri)}, the {colfull(ci)} is {sanitize(rows[ri][ci])}.' for ri, ci in facts))
        elif layout == 'F3':
            for ri, ci in facts:
                lines.append(f'- {lab(ri)} / {colfull(ci)}: {sanitize(rows[ri][ci])}')
        elif layout == 'H1':
            lines.append(' '.join(f'A {colfull(ci)} of {sanitize(rows[ri][ci])} was recorded for {lab(ri)}.' for ri, ci in facts))
        elif layout == 'H2':
            for ri, ci in facts:
                lines.append(f'{lab(ri)} comes in at {sanitize(rows[ri][ci])} in terms of {colfull(ci)}.')
        elif layout == 'H3':
            for ri, ci in facts:
                lines.append(f'The {lab(ri)} entry reports {sanitize(rows[ri][ci])} ({colfull(ci)}).')
        elif layout == 'H4':
            for ri, cis in by_row.items():
                parts = [f'{colfull(ci)} {sanitize(rows[ri][ci])}' for ci in cis]
                lines.append(f'Looking at {lab(ri)}, we see ' + ', '.join(parts[:-1]) + (', and ' if len(parts) > 1 else '') + parts[-1] + '.')
    else:
        raise ValueError(layout)
    ctx_b = [l for l in T['ctx_before'] if '|' not in l and l.strip()]
    ctx_a = [l for l in T['ctx_after'] if '|' not in l and l.strip()]
    text = '\n'.join(ctx_b + [''] + lines + [''] + ctx_a) + '\n'
    return {'text': text, 'chosen': chosen, 'ctx_numbers': sum(len(re.findall(r'\d+', l)) for l in ctx_b + ctx_a)}


if __name__ == '__main__':
    import sys
    from k3_data import load_all, usable
    sys.stdout.reconfigure(encoding='utf-8')
    tables, _ = load_all()
    U = [T for T in tables if usable(T) and T['split'] == 'confirm' and T['nnum'] >= 6][:60]
    T = [t for t in U if len(t['hk']) >= 5][0]
    for lay in LAYOUTS:
        for vf in ['v0', 'mix']:
            r = render_table(T, lay, vf, 1)
            print('=====', lay, vf)
            print(r['text'][:700] if r else None)
