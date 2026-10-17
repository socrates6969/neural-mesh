"""tokens/cost per task before/after (held | confirm), from results/h11_pipe*.json and results/h11_doc*.json (R5-C). SYNTHETIC accounting."""
import json, os, sys
R = os.path.dirname(os.path.abspath(__file__))
def L(n): return json.load(open(os.path.join(R, n)))
def comp(res, ro=4.0, ctx=1.0):
    gb, gc, go, vi, vo = res[0], res[1], res[2], res[3], res[4]
    return dict(gen_prompt_base=gb, gen_ctx=gc*ctx, gen_completion=go, ver_prompt=vi, ver_completion=vo, cost=gb+gc*ctx+ro*go+vi+ro*vo, n_gen=res[9], n_ver=res[10], n_inj=res[12], n_hit=res[13])
def show(tag, rows):
    print(tag)
    print("  %-26s %8s %8s %8s %8s %8s %8s | %5s %5s %5s" % ("arm", "base-in", "ctx-in", "compl", "ver-in", "ver-out", "COST TE", "gens", "vers", "hits"))
    for name, c in rows:
        print("  %-26s %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f | %5.2f %5.2f %5.2f" % (name, c['gen_prompt_base'], c['gen_ctx'], c['gen_completion'], c['ver_prompt'], c['ver_completion'], c['cost'], c['n_gen'], c['n_ver'], c['n_hit']))
def pipe(fn, label):
    d = L(fn)
    tw = [w for w in d['raw'] if not w['world'].startswith('ctl-')]
    for wn in ('pipe-main', 'lat-main', 'MEAN(10 treatment)', 'MEAN(5 controls)'):
        if wn.startswith('MEAN'):
            ws = tw if 'treatment' in wn else [w for w in d['raw'] if w['world'].startswith('ctl-')]
        else:
            ws = [w for w in d['raw'] if w['world'] == wn]
        rows = []
        for a in ('L2', 'L3', 'L4', 'L3+Vg', 'S'):
            acc = None
            for w in ws:
                r = w['arms'][a]['res']
                acc = r[:] if acc is None else [x + y for x, y in zip(acc, r)]
            acc = [x / len(ws) for x in acc]
            c = comp(acc)
            q = sum(w['arms'][a]['q'] for w in ws) / len(ws)
            rows.append((f"{a} (q {q:.4f})", c))
        show(f"[{label}] P {wn}", rows)
def doc(fn, label):
    d = L(fn)
    tw = [w for w in d['raw'] if w['world'] in ('sib16', 'rand16', 'sib16-mixed50', 'sib16-weakid50', 'sib16-drift', 'sib64', 'sib16-dups30')]
    cw = [w for w in d['raw'] if w['world'] in ('ctl-none', 'ctl-oneoff', 'ctl-uniq')]
    for wn, ws in (('sib16', [w for w in d['raw'] if w['world'] == 'sib16']), ('MEAN(7 treatment)', tw), ('MEAN(3 controls)', cw), ('ctl-uniq', [w for w in d['raw'] if w['world'] == 'ctl-uniq'])):
        rows = []
        for a in ('D0', 'Kstar', 'k2', 'route-CA', 'memo-ex-ver+rv', 'S-memo', 'S'):
            acc = None
            for w in ws:
                r = w['arms'][a]['res']
                acc = r[:] if acc is None else [x + y for x, y in zip(acc, r)]
            acc = [x / len(ws) for x in acc]
            c = comp(acc, ctx=4.0)
            q = sum(w['arms'][a]['succ'] for w in ws) / len(ws)
            rows.append((f"{a} (succ {q:.4f})", c))
        show(f"[{label}] D {wn}  (ctx tokens x4)", rows)
if __name__ == '__main__':
    pipe('h11_pipe.json', 'held 31-60'); pipe('h11_pipe_confirm61-90.json', 'confirm 61-90')
    doc('h11_doc.json', 'held 31-60 corpus A'); doc('h11_doc_confirm61-90.json', 'confirm 61-90 corpus B')
