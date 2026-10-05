"""R87: REAL per-agent token accounting from the recorded Claude Code transcripts on this machine (~/.claude/projects/**/*.jsonl, incl. subagents/*).
The `usage` fields are API-reported token counts (NOT proxies). AGGREGATES ONLY are computed/printed/saved: no prompt, tool or file content leaves this script
(only counts, sizes and agent-type names). Catalog (skill/agent/deferred-tool listings injected as attachments) sizes use tiktoken cl100k as a PROXY tokenizer.
Price-weighted units (ASSUMPTION, public price ratios): input 1.0, cache-write 1.25 (5m) / 2.0 (1h), cache-read 0.1, output 5.0."""
import os, json, glob, collections, statistics as st
import numpy as np, tiktoken
ENC = tiktoken.get_encoding("cl100k_base")
def ntok(s): return len(ENC.encode(s, disallowed_special=()))
def pct(a, p): return float(np.percentile(a, p)) if len(a) else 0.0
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
def parse(f):
    msgs = {}; cat = collections.Counter(); tools = collections.Counter(); first_cat = {}
    for line in open(f, encoding="utf-8", errors="ignore"):
        try: o = json.loads(line)
        except Exception: continue
        t = o.get("type")
        if t == "assistant":
            m = o.get("message", {}); u = m.get("usage")
            if not u: continue
            mid = m.get("id") or o.get("uuid"); prev = msgs.get(mid)
            if prev is None: msgs[mid] = dict(u=u, ts=o.get("timestamp"))
            elif u.get("output_tokens", 0) >= prev["u"].get("output_tokens", 0): prev["u"] = u
            for c in m.get("content", []):
                if c.get("type") == "tool_use": tools[c.get("name")] += 1
        elif t == "attachment":
            a = o.get("attachment", {}); at = a.get("type")
            if at == "skill_listing": txt = a.get("content") or ""
            elif at in ("agent_listing_delta", "deferred_tools_delta"): txt = "\n".join(a.get("addedLines") or [])
            else: continue
            if at not in first_cat or a.get("isInitial"): first_cat[at] = first_cat.get(at, 0)
            cat[at + "_chars"] += len(txt); cat[at + "_n"] += 1
            if txt and cat[at + "_n"] == 1: cat[at + "_tok_proxy"] += ntok(txt)
    return msgs, cat, tools
rows = []
for f in glob.glob(os.path.expanduser("~/.claude/projects/*/**/*.jsonl"), recursive=True):
    p = f.replace("\\", "/"); sub = "/subagents/" in p; atype = "main"
    if sub:
        try: atype = json.load(open(f[:-6] + ".meta.json")).get("agentType", "unknown")
        except Exception: atype = "unknown"
    msgs, cat, tools = parse(f)
    if not msgs: continue
    order = sorted(msgs.values(), key=lambda x: x["ts"] or ""); us = [m["u"] for m in order]
    ctx = [u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0) for u in us]
    cw5 = sum((u.get("cache_creation") or {}).get("ephemeral_5m_input_tokens", 0) for u in us); cw1 = sum((u.get("cache_creation") or {}).get("ephemeral_1h_input_tokens", 0) for u in us)
    cwt = sum(u.get("cache_creation_input_tokens", 0) for u in us); cw5 += max(0, cwt - cw5 - cw1)
    inp = sum(u.get("input_tokens", 0) for u in us); cr = sum(u.get("cache_read_input_tokens", 0) for u in us); out = sum(u.get("output_tokens", 0) for u in us)
    price = inp + 1.25 * cw5 + 2.0 * cw1 + 0.1 * cr + 5.0 * out
    rows.append(dict(kind="subagent" if sub else "main", atype=atype, calls=len(us), first_ctx=ctx[0], peak_ctx=max(ctx), inp=inp, cw=cwt, cr=cr, out=out, price=price,
                     cat_tok=cat["skill_listing_tok_proxy"] + cat["agent_listing_delta_tok_proxy"] + cat["deferred_tools_delta_tok_proxy"], cat_skill=cat["skill_listing_tok_proxy"], cat_agent=cat["agent_listing_delta_tok_proxy"], cat_def=cat["deferred_tools_delta_tok_proxy"],
                     n_tools=sum(tools.values())))
print("files with usage:", len(rows), " main:", sum(r["kind"] == "main" for r in rows), " subagent:", sum(r["kind"] == "subagent" for r in rows))
def summ(rs, name):
    if not rs: return None
    s = dict(name=name, n=len(rs), calls_med=pct([r["calls"] for r in rs], 50), first_ctx_med=pct([r["first_ctx"] for r in rs], 50), first_ctx_p90=pct([r["first_ctx"] for r in rs], 90),
             total_med=pct([r["inp"] + r["cw"] + r["cr"] + r["out"] for r in rs], 50), out_med=pct([r["out"] for r in rs], 50), price_med=pct([r["price"] for r in rs], 50), price_sum=sum(r["price"] for r in rs),
             tok_sum=sum(r["inp"] + r["cw"] + r["cr"] + r["out"] for r in rs), out_sum=sum(r["out"] for r in rs), cr_sum=sum(r["cr"] for r in rs), cw_sum=sum(r["cw"] for r in rs), in_sum=sum(r["inp"] for r in rs),
             cat_med=pct([r["cat_tok"] for r in rs], 50), cat_with=sum(1 for r in rs if r["cat_tok"] > 0))
    return s
out = {"overall": {k: summ([r for r in rows if r["kind"] == k], k) for k in ("main", "subagent")}, "by_agent_type": {}}
for k, s in out["overall"].items(): print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in s.items()})
by = collections.defaultdict(list)
for r in rows:
    if r["kind"] == "subagent": by[r["atype"]].append(r)
print(f"{'agentType':34s} {'n':>4s} {'calls':>6s} {'firstctx':>9s} {'total_med':>10s} {'out_med':>8s} {'priceU_med':>10s} {'share_price%':>12s}")
tp = sum(r["price"] for rs in by.values() for r in rs)
for at, rs in sorted(by.items(), key=lambda x: -sum(r["price"] for r in x[1]))[:18]:
    s = summ(rs, at); out["by_agent_type"][at] = s
    print(f"{at:34s} {s['n']:4d} {s['calls_med']:6.0f} {s['first_ctx_med']:9.0f} {s['total_med']:10.0f} {s['out_med']:8.0f} {s['price_med']:10.0f} {100*s['price_sum']/tp:12.1f}")
# catalog share of the first-call context (proxy / API-reported)
for k in ("main", "subagent"):
    rs = [r for r in rows if r["kind"] == k and r["cat_tok"] > 0]
    if rs:
        sh = [min(1.0, r["cat_tok"] / r["first_ctx"]) for r in rs if r["first_ctx"] > 0]
        print(f"{k}: {len(rs)} files carry catalog attachments; catalog tokens (cl100k proxy, FIRST listing of each kind) median {pct([r['cat_tok'] for r in rs],50):.0f}  [skills {pct([r['cat_skill'] for r in rs],50):.0f}, agents {pct([r['cat_agent'] for r in rs],50):.0f}, deferred tools {pct([r['cat_def'] for r in rs],50):.0f}]; share of API-reported first-call context median {pct(sh,50):.3f}, p90 {pct(sh,90):.3f}")
    else: print(k, "no catalog attachments recorded in these files")
json.dump(out, open(os.path.join(RES, "r87.json"), "w"), indent=1, default=float)

# ---- R87b: what-if from REAL numbers: a lean (explicit narrow tool list) agent has first-call context ~18k (ruflo-agent:nested-queen, n=5) vs ~95k for all-tools agents.
lean = [r for r in rows if r["kind"] == "subagent" and r["first_ctx"] < 40000]; fat = [r for r in rows if r["kind"] == "subagent" and r["first_ctx"] >= 40000]
L = float(np.median([r["first_ctx"] for r in lean])) if lean else None
print(f"\nlean agents (first ctx < 40k): n={len(lean)} median first ctx {L}; all-tools agents n={len(fat)} median {pct([r['first_ctx'] for r in fat],50):.0f}")
if lean:
    sav_p = sav_t = 0.0
    for r in fat:
        d = r["first_ctx"] - L; sav_t += d * r["calls"]; sav_p += d * (2.0 + 0.1 * (r["calls"] - 1))
    P = sum(r["price"] for r in fat); T = sum(r["inp"] + r["cw"] + r["cr"] + r["out"] for r in fat)
    print(f"what-if all-tools -> lean first-call context ({L:.0f} tok) for the same calls: raw tokens saved {sav_t/T:.3f} of {T/1e6:.0f}M; price-weighted units saved {sav_p/P:.3f}")
    out["whatif_lean"] = dict(lean_n=len(lean), lean_first_ctx=L, fat_n=len(fat), raw_saved_frac=sav_t / T, price_saved_frac=sav_p / P)
tiny = [r for r in rows if r["kind"] == "subagent" and r["calls"] <= 2]
print(f"spawns with <=2 API calls: {len(tiny)} of {sum(r['kind']=='subagent' for r in rows)}; their share of subagent price units {sum(r['price'] for r in tiny)/sum(r['price'] for r in rows if r['kind']=='subagent'):.4f}; median output tokens {pct([r['out'] for r in tiny],50):.0f}")
cs = [r["calls"] for r in rows if r["kind"] == "subagent"]
print("subagent calls per spawn: p10/p50/p90 =", pct(cs, 10), pct(cs, 50), pct(cs, 90), "; price decomposition (all subagents): cache-read units", f"{0.1*sum(r['cr'] for r in rows if r['kind']=='subagent')/sum(r['price'] for r in rows if r['kind']=='subagent'):.3f}",
      "cache-write", f"{(sum(r['price'] for r in rows if r['kind']=='subagent') - 0.1*sum(r['cr'] for r in rows if r['kind']=='subagent') - 5*sum(r['out'] for r in rows if r['kind']=='subagent') - sum(r['inp'] for r in rows if r['kind']=='subagent'))/sum(r['price'] for r in rows if r['kind']=='subagent'):.3f}", "output", f"{5*sum(r['out'] for r in rows if r['kind']=='subagent')/sum(r['price'] for r in rows if r['kind']=='subagent'):.3f}")
json.dump(out, open(os.path.join(RES, "r87.json"), "w"), indent=1, default=float)
