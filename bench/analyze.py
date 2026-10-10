"""bench/analyze.py - validate and analyse bench/results/<tag>/raw.jsonl (one or more tags; HELD and CONF may share a tag).

Validation (split-aware rows written by run.py):
  * exactly one final row (status ok | model_fail) per (split, id, cond, rep), probe rows excluded - else exit 4;
  * every pair complete: world claude = A vs B on every rep 0..K-1, world local = C vs D on rep 0 - else exit 3, or with
    --drop-incomplete the item is dropped from that world (reported).
Item score = mean correctness over its reps (model_fail counts as wrong; infra_fail rows are never scored).

HELD/CONF verdict: the verify-before-keep gate of py/b1 (gate.compare, lib.holm/lib.tost - reused, not reimplemented). Per
(axis x world x split) a paired test of the item scores, item i of an axis sitting in seed slot 31+i (HELD) or 61+i (CONF);
worlds: A = claude (A vs B), B = local (C vs D). Holm over the whole family. KEEP only if the target axis research_private
(the "librarian" claim) gains Holm-significantly in the same world in BOTH HELD and CONF and no axis (math, physics, code,
research_public are the guards) has a Holm-significant loss beyond tol; TOST +-delta is reported as equivalence evidence.
tol/delta come from PREREG.json. Tokens and latency are reported, never gated. TUNE rows are described with the paired item
bootstrap only (no verdict). Legacy rows (v1/v2, no status field) get the old descriptive table.

Writes results/<first tag>/summary.json and summary.txt (with a paste-ready RD-LOG line).
Usage: python bench/analyze.py TAG [TAG ...] [--drop-incomplete] [--json OUT]
HELD/CONF manifests must carry the prereg_sha of the current bench/PREREG.json (the registration the run was checked
against); a mismatch exits 2. A TUNE mismatch is only a warning line (TUNE is descriptive).
Exit: 0 ok, 2 PREREG mismatch, 3 incomplete pairs, 4 invalid rows (duplicates / unknown items / battery mismatch)."""
import argparse, collections, json, os, pathlib, random, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "py" / "b1"))         # reuse the B1 gate as r115.py does; appended so bench/battery.py wins over py/b1/battery.py
from lib import holm, tost, paired, HELD, CONF  # noqa: E402,F401
import gate  # noqa: E402
from common import AXES, axis_of, battery_path, json_sha, load_prereg  # noqa: E402

RESULTS = pathlib.Path(os.environ.get("BENCH_RESULTS") or HERE / "results")
EXIT_OK, EXIT_PREREG, EXIT_INCOMPLETE, EXIT_INVALID = 0, 2, 3, 4
WORLDS = {"A": ("claude", "A", "B"), "B": ("local", "C", "D")}       # gate world key -> (name, base cond, candidate cond)
SLOT0 = {"held": HELD[0], "conf": CONF[0]}                          # 31 / 61
TARGET = ["research_private"]
DEFAULT_TOL, DEFAULT_DELTA = 0.002, 0.01


class Invalid(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


def boot(d, n=10000, seed=1):
    if not d: return (float("nan"),) * 3
    rng = random.Random(seed); m = sum(d) / len(d)
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n))
    return m, ms[int(.025 * n)], ms[int(.975 * n)]


# ------------------------------------------------------------------ legacy (v1/v2/hook1 rows: no status field)
def legacy_describe(rows):
    by = collections.defaultdict(dict)
    for r in rows: by[r["cond"]][r["id"]] = r
    cats = ["math", "physics", "code", "research", "research:public", "research:private", "ALL", "ALL_excl_private"]

    def sel(r, cat):
        if cat == "ALL": return True
        if cat == "ALL_excl_private": return not (r["cat"] == "research" and r["sub"] == "private")
        if ":" in cat: c, s = cat.split(":"); return r["cat"] == c and r["sub"] == s
        return r["cat"] == cat

    def val(r, m):
        if m == "acc": return float(r["correct"])
        if m == "complete": return float(r["completed"])
        if m == "parsed": return float(r.get("parsed", False))
        if m == "tokens": return float(r["tokens"]["total"]) if r.get("tokens") else None
        if m == "wall": return r["wall_s"]
        if m == "api_s": return (r["api_ms"] or 0) / 1000 if r.get("api_ms") is not None else None
    out = {}
    for cat in cats:
        out[cat] = {}
        for cond in sorted(by):
            rs = [r for r in by[cond].values() if sel(r, cat)]
            if not rs: continue
            o = {"n": len(rs)}
            for m in ("acc", "complete", "parsed", "tokens", "wall", "api_s"):
                v = [x for x in (val(r, m) for r in rs) if x is not None]
                o[m] = sum(v) / len(v) if v else None
            out[cat][cond] = o
        for cond in sorted(by):
            if cond == "A" or "A" not in by: continue
            ids = [i for i in by[cond] if i in by["A"] and sel(by[cond][i], cat)]
            if not ids: continue
            pr = {}
            for m in ("acc", "tokens", "wall", "api_s"):
                d = [val(by[cond][i], m) - val(by["A"][i], m) for i in ids if val(by["A"][i], m) is not None and val(by[cond][i], m) is not None]
                pr[m] = boot(d)
            out[cat].setdefault("paired_vs_A", {})[cond] = {"n": len(ids), **{m: [round(x, 4) for x in v] for m, v in pr.items()}}
    f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
    lines = []
    for cat in cats:
        if not out[cat]: continue
        lines.append(f"== {cat}")
        for cond in sorted(by):
            o = out[cat].get(cond)
            if o: lines.append(f"  {cond}: n={o['n']:3d} acc={f(o['acc'])} complete={f(o['complete'])} parsed={f(o['parsed'])} tok={f(o['tokens'], 0)} wall_s={f(o['wall'], 1)} api_s={f(o['api_s'], 1)}")
        for cond, p in out[cat].get("paired_vs_A", {}).items():
            lines.append(f"  {cond}-A (n={p['n']}): acc {p['acc'][0]:+.3f} [{p['acc'][1]:+.3f},{p['acc'][2]:+.3f}]  tokens {p['tokens'][0]:+.0f} [{p['tokens'][1]:+.0f},{p['tokens'][2]:+.0f}]  "
                         f"wall {p['wall'][0]:+.1f}s [{p['wall'][1]:+.1f},{p['wall'][2]:+.1f}]  api {p['api_s'][0]:+.1f}s [{p['api_s'][1]:+.1f},{p['api_s'][2]:+.1f}]")
    return out, lines


# ------------------------------------------------------------------ validation
def load_rows(tags):
    rows = []
    for t in tags:
        p = RESULTS / t / "raw.jsonl"
        if not p.exists(): raise Invalid(EXIT_INVALID, f"{p} missing")
        for l in open(p, encoding="utf8"):
            if l.strip(): rows.append(dict(json.loads(l), _tag=t))
    return rows


def manifests(tags):
    out = {}
    for t in tags:
        for s in ("tune", "held", "conf"):
            p = RESULTS / t / f"manifest_{s}.json"
            if p.exists(): out[s] = json.load(open(p, encoding="utf8"))
    return out


def validate(rows, mans, drop_incomplete):
    """-> per split: dict(items=[...in battery order], scores={cond: {id: mean}}, kept={world: [ids]}, dropped, reps)"""
    final = [r for r in rows if r.get("status") in ("ok", "model_fail") and not r.get("probe")]
    cnt = collections.Counter((r["split"], r["id"], r["cond"], r.get("rep", 0)) for r in final)
    dupes = [k for k, v in cnt.items() if v > 1]
    if dupes: raise Invalid(EXIT_INVALID, f"{len(dupes)} units have more than one final row, e.g. {dupes[:3]}")
    out = {}
    for split in sorted({r["split"] for r in rows}):
        man = mans.get(split)
        if man is None: raise Invalid(EXIT_INVALID, f"manifest_{split}.json missing")
        bp = battery_path(split)
        if not bp.exists(): raise Invalid(EXIT_INVALID, f"{bp.name} missing")
        items = json.load(open(bp, encoding="utf8"))["items"]
        if json_sha(items) != man["battery_sha"]: raise Invalid(EXIT_INVALID, f"{bp.name} is not the battery {split} was run on (items_sha differs)")
        known = {i["id"] for i in items}
        rs = [r for r in final if r["split"] == split]
        unknown = {r["id"] for r in rows if r.get("split") == split} - known
        if unknown: raise Invalid(EXIT_INVALID, f"{split}: rows for unknown items {sorted(unknown)[:5]}")
        conds = man["conds"]; K = man["K"]["claude"]
        have = {(r["id"], r["cond"], r.get("rep", 0)): r for r in rs}
        if split == "tune":      # TUNE may be run on a subset (--limit/--cats): the planned set is every item with any row
            ran = {r["id"] for r in rows if r.get("split") == split}
            items = [i for i in items if i["id"] in ran]
        kept, dropped, missing = {}, {}, []
        for wkey, (wname, bc, cc) in WORLDS.items():
            if bc not in conds or cc not in conds: continue
            reps = range(K) if wname == "claude" else range(1)
            ok_ids = []
            for it in items:
                miss = [(it["id"], c, k) for c in (bc, cc) for k in reps if (it["id"], c, k) not in have]
                if miss: missing += miss; dropped.setdefault(wname, []).append(it["id"])
                else: ok_ids.append(it["id"])
            kept[wname] = ok_ids
        if missing and not drop_incomplete:
            raise Invalid(EXIT_INCOMPLETE, f"{split}: {len(missing)} units missing for complete pairs (e.g. {missing[:3]}); re-run run.py or pass --drop-incomplete")
        scores = collections.defaultdict(dict)
        for (iid, c, k), r in have.items():
            scores[c].setdefault(iid, []).append(float(bool(r.get("correct"))))
        scores = {c: {i: sum(v) / len(v) for i, v in d.items()} for c, d in scores.items()}
        out[split] = dict(items=items, scores=scores, kept=kept, dropped=dropped, K=K, conds=conds, rows=rs, manifest=man,
                          probe_rows=[r for r in rows if r.get("split") == split and r.get("probe") and r.get("status") in ("ok", "model_fail")])
    return out


# ------------------------------------------------------------------ analysis
def gate_tables(v):
    """base/cand dicts for gate.compare: (world, seed slot) -> {axis: item score}"""
    base, cand, slots = {}, {}, {}
    for split in ("held", "conf"):
        if split not in v: continue
        S = v[split]
        for wkey, (wname, bc, cc) in WORLDS.items():
            if wname not in S["kept"]: continue
            keep = set(S["kept"][wname])
            for ax in AXES:
                ax_items = [i for i in S["items"] if axis_of(i) == ax]        # battery order defines the seed slot
                for idx, it in enumerate(ax_items):
                    if it["id"] not in keep: continue
                    seed = SLOT0[split] + idx
                    base.setdefault((wkey, seed), {})[ax] = S["scores"][bc][it["id"]]
                    cand.setdefault((wkey, seed), {})[ax] = S["scores"][cc][it["id"]]
                    slots[(split, it["id"])] = seed
    return base, cand, slots


def per_cond_costs(S):
    out = {}
    for c in S["conds"]:
        rs = [r for r in S["rows"] if r["cond"] == c and r.get("status") in ("ok", "model_fail") and not r.get("probe")]
        if not rs: continue
        tok = [r["tokens"]["total"] for r in rs if r.get("tokens")]
        wall = [r["wall_s"] for r in rs if r.get("wall_s") is not None]
        cost = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
        out[c] = dict(n=len(rs), acc=sum(bool(r.get("correct")) for r in rs) / len(rs), fallback_acc=sum(bool(r.get("fallback_correct")) for r in rs) / len(rs),
                      parsed=sum(bool(r.get("parsed")) for r in rs) / len(rs), model_fail=sum(r["status"] == "model_fail" for r in rs),
                      tokens=sum(tok) / len(tok) if tok else None, wall_s=sum(wall) / len(wall) if wall else None, cost_usd=sum(cost) if cost else None,
                      mean_attempts=sum(r.get("attempts", 1) for r in rs) / len(rs), model_ids=sorted({m for r in rs for m in (r.get("model_ids") or [])}))
    return out


def paired_costs(S):
    """per world: paired item bootstrap of candidate - base for tokens and wall-clock (reported only, never gated)"""
    out = {}
    for wkey, (wname, bc, cc) in WORLDS.items():
        if wname not in S["kept"]: continue
        keep = set(S["kept"][wname])
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in S["rows"]:
            if r["id"] in keep and r["cond"] in (bc, cc) and r.get("status") in ("ok", "model_fail") and not r.get("probe"):
                if r.get("tokens"): agg[("tok", r["cond"])][r["id"]].append(r["tokens"]["total"])
                if r.get("wall_s") is not None: agg[("wall", r["cond"])][r["id"]].append(r["wall_s"])
        res = {}
        for m in ("tok", "wall"):
            a, b = agg[(m, bc)], agg[(m, cc)]
            d = [sum(b[i]) / len(b[i]) - sum(a[i]) / len(a[i]) for i in keep if a.get(i) and b.get(i)]
            res[m] = [round(x, 3) for x in boot(d)] + [len(d)]
        out[wname] = res
    return out


def describe_split(S):
    """paired item bootstrap of cand - base accuracy per axis and world (TUNE: the only analysis; HELD/CONF: context)"""
    out = {}
    for wkey, (wname, bc, cc) in WORLDS.items():
        if wname not in S["kept"]: continue
        keep = set(S["kept"][wname])
        for ax in list(AXES) + ["ALL"]:
            ids = [i["id"] for i in S["items"] if i["id"] in keep and (ax == "ALL" or axis_of(i) == ax)]
            if not ids: continue
            d = [S["scores"][cc][i] - S["scores"][bc][i] for i in ids]
            m, lo, hi = boot(d)
            out.setdefault(wname, {})[ax] = dict(n=len(ids), base=sum(S["scores"][bc][i] for i in ids) / len(ids), cand=sum(S["scores"][cc][i] for i in ids) / len(ids),
                                                 diff=round(m, 4), ci=[round(lo, 4), round(hi, 4)])
    return out


def determinism(S):
    out = {}
    rows = S["rows"] + S["probe_rows"]
    for c in ("C", "D"):
        probe = {r["id"]: r for r in rows if r["cond"] == c and r.get("probe")}
        base = {r["id"]: r for r in rows if r["cond"] == c and not r.get("probe") and r.get("rep", 0) == 0}
        both = [i for i in probe if i in base]
        if both:
            out[c] = dict(n=len(both), identical_text=sum(probe[i]["text"] == base[i]["text"] for i in both),
                          same_correct=sum(bool(probe[i].get("correct")) == bool(base[i].get("correct")) for i in both))
    return out


def analyse(v, tags, prereg, prereg_sha):
    tol = (prereg or {}).get("tol", DEFAULT_TOL); delta = (prereg or {}).get("delta", DEFAULT_DELTA)
    summary = dict(tags=tags, prereg_sha=prereg_sha, tol=tol, delta=delta, target=TARGET, guards=[a for a in AXES if a not in TARGET], splits={})
    for split, S in v.items():
        summary["splits"][split] = dict(n_items=len(S["items"]), K=S["K"], conds=S["conds"], dropped=S["dropped"], per_cond=per_cond_costs(S),
                                        paired_costs=paired_costs(S), describe=describe_split(S), determinism=determinism(S),
                                        manifest={k: S["manifest"].get(k) for k in ("git_sha", "git_dirty", "battery_sha", "corpus_sha", "frontend_sha", "prereg_sha", "sandbox", "models")})
    lines = []
    if "held" in v or "conf" in v:
        base, cand, slots = gate_tables(v)
        verdict, rows, gain, loss = gate.compare(base, cand, TARGET, tol=tol, delta=delta, quiet=True, axes=list(AXES))
        both = "held" in v and "conf" in v
        summary["gate"] = dict(verdict=verdict if both else f"PARTIAL ({'held' if 'held' in v else 'conf'} only; the gate needs both HELD and CONF)",
                               complete=both, holm_family=len(rows), gains=gain, losses=loss,
                               tests=[dict(axis=r["ax"], world=WORLDS[r["w"]][0], split=r["sp"], diff=round(r["mean"], 4), ci=[round(r["lo"], 4), round(r["hi"], 4)],
                                           p=r["p"], p_holm=r["ph"], tost_p=r["tost"], n=r["n"], wins=r["wins"], losses=r["losses"]) for r in rows])
        g = summary["gate"]
        lines.append(f"== GATE (py/b1/gate.compare; target {TARGET}, guards {summary['guards']}, tol {tol}, TOST +-{delta}, Holm family {len(rows)})")
        for t in g["tests"]:
            lines.append(f"  {t['axis']:17s} {t['world']:6s} {t['split']}: d={t['diff']:+.4f} [{t['ci'][0]:+.4f},{t['ci'][1]:+.4f}] n={t['n']} p_holm={t['p_holm']:.3g} TOST_p={t['tost_p']:.3g} W/L={t['wins']}/{t['losses']}")
        lines.append(f"  target gains (Holm, both splits): {[x for x in gain if x[0] in TARGET] or 'none'}")
        lines.append(f"  Holm losses: {loss or 'none'}")
        lines.append(f"  VERDICT: {g['verdict']}")
    for split, s in summary["splits"].items():
        lines.append(f"== {split.upper()} (n_items={s['n_items']}, K={s['K']}, conds={','.join(s['conds'])}){' DESCRIPTIVE ONLY' if split == 'tune' else ''}")
        for w, axs in s["describe"].items():
            for ax, d in axs.items():
                lines.append(f"  {w:6s} {ax:17s} n={d['n']:3d} base={d['base']:.3f} cand={d['cand']:.3f} diff={d['diff']:+.3f} [{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}]")
        for c, o in s["per_cond"].items():
            lines.append(f"  {c}: rows={o['n']} acc={o['acc']:.3f} (fallback {o['fallback_acc']:.3f}, diagnostic) parsed={o['parsed']:.3f} model_fail={o['model_fail']} "
                         f"tokens={o['tokens'] if o['tokens'] is None else round(o['tokens'])} wall_s={o['wall_s'] if o['wall_s'] is None else round(o['wall_s'], 1)} "
                         f"cost=${o['cost_usd'] or 0:.2f} attempts={o['mean_attempts']:.2f}")
        for w, pc in s["paired_costs"].items():
            lines.append(f"  {w} cost (reported, NOT gated): tokens {pc['tok'][0]:+.0f} [{pc['tok'][1]:+.0f},{pc['tok'][2]:+.0f}] wall {pc['wall'][0]:+.1f}s [{pc['wall'][1]:+.1f},{pc['wall'][2]:+.1f}] (n={pc['tok'][3]})")
        if s["determinism"]: lines.append(f"  determinism probe: {s['determinism']}")
        if s["dropped"]: lines.append(f"  dropped (incomplete pairs): { {w: len(x) for w, x in s['dropped'].items()} }")
    summary["rd_log_line"] = rd_log_line(summary, tags)
    lines.append("RD-LOG: " + summary["rd_log_line"])
    return summary, lines


def check_prereg(v, prereg_sha):
    """HELD/CONF: the manifest's prereg_sha must be the current PREREG.json sha (exit 2); TUNE: a mismatch is a warning"""
    warns = []
    for split, S in v.items():
        got = S["manifest"].get("prereg_sha")
        if got == prereg_sha: continue
        msg = f"{split}: manifest prereg_sha {str(got)[:12]} != current PREREG.json {str(prereg_sha)[:12]}"
        if split in ("held", "conf"): raise Invalid(EXIT_PREREG, msg + " (the run was checked against a different registration)")
        warns.append(msg)
    return warns


def rd_log_line(summary, tags):
    pre = (summary.get("prereg_sha") or "none")[:12]
    if "gate" not in summary:
        parts = []
        for split, s in summary["splits"].items():
            for w, axs in s["describe"].items():
                if "ALL" in axs: parts.append(f"{split}/{w} ALL {axs['ALL']['diff']:+.3f} [{axs['ALL']['ci'][0]:+.3f},{axs['ALL']['ci'][1]:+.3f}]")
        return f"Bench battery ({'+'.join(tags)}): TUNE-only descriptive evidence, no verdict; " + "; ".join(parts)
    g = summary["gate"]
    tgt = [t for t in g["tests"] if t["axis"] in TARGET]
    tdesc = ", ".join(f"{t['world']} {t['split']} d={t['diff']:+.3f} p_holm={t['p_holm']:.2g}" for t in tgt) or "no target tests"
    costs = []
    for split in ("held", "conf"):
        for w, pc in summary["splits"].get(split, {}).get("paired_costs", {}).items():
            costs.append(f"{split}/{w} tokens {pc['tok'][0]:+.0f}")
    return (f"Bench battery HELD/CONF ({'+'.join(tags)}, PREREG sha {pre}): verdict {g['verdict']}; target research_private: {tdesc}; "
            f"Holm losses: {g['losses'] or 'none'} (family {g['holm_family']}, tol {summary['tol']}, TOST +-{summary['delta']}); cost reported not gated: {', '.join(costs) or 'n/a'}.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="analyze.py")
    ap.add_argument("tags", nargs="+"); ap.add_argument("--drop-incomplete", action="store_true"); ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    try:
        rows = load_rows(a.tags)
        out_dir = RESULTS / a.tags[0]
        if rows and all("status" not in r for r in rows):          # v1/v2/hook1 rows
            summary, lines = legacy_describe(rows)
        else:
            v = validate([r for r in rows if "status" in r], manifests(a.tags), a.drop_incomplete)
            prereg, prereg_sha = load_prereg()
            warns = check_prereg(v, prereg_sha)
            summary, lines = analyse(v, a.tags, prereg, prereg_sha)
            summary["prereg_warnings"] = warns
            lines = [f"warning: {w}" for w in warns] + lines
    except Invalid as e:
        print(f"analyze.py: {e}", file=sys.stderr); return e.code
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf8"), indent=1, default=str)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
    if a.json: json.dump(summary, open(a.json, "w", encoding="utf8"), indent=1, default=str)
    print("\n".join(lines))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
