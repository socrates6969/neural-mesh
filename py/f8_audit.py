"""f8 audit: structural scan of an ONNX file WITHOUT executing it and without the onnx package.
Minimal protobuf wire-format walker: lists opset domains, node op_types/domains, initializer data_location
(external data), metadata_props, and flags anything outside the standard ai.onnx domain.
Imports only json/sys/hashlib/collections. No network. Usage: python py/f8_audit.py <model.onnx> <tokenizer.json>"""
import hashlib, json, sys, collections

def rd(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7F) << s
        if c < 0x80: return r, i
        s += 7

def fields(b, a=0, e=None):
    e = len(b) if e is None else e; i = a
    while i < e:
        k, i = rd(b, i); f, w = k >> 3, k & 7
        if w == 0: v, i = rd(b, i); yield f, w, v
        elif w == 2: n, i = rd(b, i); yield f, w, (i, i + n); i += n
        elif w == 1: yield f, w, (i, i + 8); i += 8
        elif w == 5: yield f, w, (i, i + 4); i += 4
        else: raise ValueError("bad wire type %d" % w)

def s(b, r): return bytes(b[r[0]:r[1]]).decode("utf8", "replace")

def main():
    mp, tp = sys.argv[1], sys.argv[2]
    b = memoryview(open(mp, "rb").read())
    h = hashlib.sha256(b).hexdigest()
    out = dict(model=mp, bytes=len(b), sha256=h, opsets=[], producer=None, ops=collections.Counter(), node_domains=collections.Counter(),
               n_initializers=0, external_initializers=0, metadata_props=[], functions=0, subgraph_attrs=0)
    for f, w, v in fields(b):
        if f == 2: out["producer"] = s(b, v)
        elif f == 8:
            d = dict(domain="", version=None)
            for ff, ww, vv in fields(b, v[0], v[1]):
                if ff == 1: d["domain"] = s(b, vv)
                elif ff == 2: d["version"] = vv
            out["opsets"].append(d)
        elif f == 14:
            kv = {}
            for ff, ww, vv in fields(b, v[0], v[1]):
                if ff in (1, 2): kv[ff] = s(b, vv)
            out["metadata_props"].append(kv)
        elif f == 25: out["functions"] += 1
        elif f == 7:
            for gf, gw, gv in fields(b, v[0], v[1]):
                if gf == 1:  # node
                    op, dom = None, ""
                    for nf, nw, nv in fields(b, gv[0], gv[1]):
                        if nf == 4: op = s(b, nv)
                        elif nf == 7: dom = s(b, nv)
                        elif nf == 5:
                            for af, aw, av in fields(b, nv[0], nv[1]):
                                if af in (6, 11) and aw == 2: out["subgraph_attrs"] += 1
                    out["ops"][op] += 1; out["node_domains"][dom or "ai.onnx"] += 1
                elif gf == 5:  # initializer
                    out["n_initializers"] += 1
                    for tf, tw, tv in fields(b, gv[0], gv[1]):
                        if tf == 14 and tv == 1: out["external_initializers"] += 1
                        if tf == 13: out["external_initializers"] += 1
    t = json.load(open(tp, encoding="utf8"))
    tok = dict(sha256=hashlib.sha256(open(tp, "rb").read()).hexdigest(), keys=list(t.keys()), normalizer=t["normalizer"]["type"],
               pre_tokenizer=t["pre_tokenizer"]["type"], model=t["model"]["type"], vocab=len(t["model"]["vocab"]),
               post=t["post_processor"]["type"], decoder=(t.get("decoder") or {}).get("type"))
    out["ops"] = dict(out["ops"]); out["node_domains"] = dict(out["node_domains"])
    bad = [d for d in out["opsets"] if d["domain"] not in ("", "ai.onnx")] + [d for d in out["node_domains"] if d not in ("ai.onnx",)]
    out["VERDICT"] = "CLEAN: only standard ai.onnx ops, no external data, no functions" if not bad and not out["external_initializers"] and not out["functions"] else "SUSPICIOUS %s" % bad
    print(json.dumps(dict(onnx=out, tokenizer=tok), indent=1))

if __name__ == "__main__":
    main()
