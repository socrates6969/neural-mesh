"""Front-end replay evaluation v2 (`python -m llm frontend-eval`, writes models/frontend_eval.json schema 2). Iron rules:

* units = ingestor clusters (sft_meta.cluster; a row without meta falls back to its session). Rows the production model was TRAINED on
  are dropped together with every row of their cluster (no paraphrase of a trained row is replayed).
* the remaining units, ordered by their first ts, are cut into three CONTIGUOUS thirds: TUNE (thresholds), HELD and CONF (two
  independent confirmations). A unit never straddles two splits.
* rolling origin: a replayed row only sees rows with ts < its own ts (cache bank, OOD similarity, retrieved context), minus rows of
  the same session within SAME_EVENT_S (possible duplicate records of the same exchange). Earlier same-cluster rows outside that window
  stay: a genuine earlier repeat is exactly what the live cache / retrieval sees.
* worlds = the row `source` (agent_report, human prompt, ...): a claim must hold within one world on HELD and on CONF.
* gates per split: n >= 30, sessions >= 3, clusters >= 3, judge hard-decoy reject >= .70, worlds >= 2. Any miss -> INSUFFICIENT (the
  structural gates are checked BEFORE any model call).
* the grid (min-conf in [-1, -0.02], min-sim in [.3, .7]; no -99 / 0 sentinels) is tuned on TUNE only: max saved tokens s.t. wrong rate
  of served answers <= .10 and >= 10 local answers; no such point -> REJECTED.
* HELD and CONF go through the Rust policy. Tests per (split, world), all in ONE Holm family: saved tokens > 0 (cluster bootstrap,
  one-sided), quality loss < 0 (one-sided), TOST +-0.01 on the quality delta. Quality delta per row = judge score of the served
  answer - judge score of Claude's own answer for that row (the forward baseline, fwd_score); 0 for a forwarded row. KEEP iff saved
  tokens are Holm-significant in the SAME world on HELD and CONF, no Holm-significant quality loss, Holm-significant TOST
  equivalence (+-0.01) of the quality delta in EVERY (split, world), Wilson 95% upper bound of the wrong rate <= .10 on HELD and
  CONF, and no Rust call failed. ONLY a KEEP writes models/frontend_cfg.json (thresholds only; enabling stays an explicit --enabled).

Every model touch goes through a Backend (RealBackend: Qwen tokenizer, MiniLM, Ollama local model, local judge); tests pass a stub."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import config, data, stats
from .frontend import (SAME_EVENT_S, TARGET_WRONG, WRONG, _sim, binary_fresh, chunk_text, parse_ts, rec_signals, rust_decide,
                       verified_cache_hit)

SCHEMA = 2
GATES = {"n": 30, "sessions": 3, "clusters": 3, "decoy_reject": 0.70, "worlds": 2}
GRID_CONF = (-1.0, -0.7, -0.5, -0.3, -0.2, -0.15, -0.1, -0.05, -0.02)
GRID_SIM = (0.3, 0.4, 0.5, 0.6, 0.7)
MIN_LOCAL_TUNE = 10
WRONG_UCB_MAX = 0.10
ALPHA = 0.05
TOST_EPS = 0.01
REPS = 4000
TOP_CHUNKS = 8
MAX_LOCAL_TOKENS = 400
CTX_BUDGET = 800


def unit_of(r: dict, meta: dict) -> str:
    c = (meta.get(r["id"]) or {}).get("cluster")
    return f"c{c}" if c is not None else f"s:{r.get('session') or r['id']}"


def cluster_split(rows: list[dict], meta: dict, trained_ids) -> dict:
    """{tune, held, conf: [rows sorted by ts], dropped_ids, dropped_units, units: {split: [unit ids in order]}}"""
    trained = set(trained_ids or ())
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(unit_of(r, meta), []).append(r)
    bad = {u for u, rs in by.items() if any(r["id"] in trained for r in rs)}
    keep = sorted((u for u in by if u not in bad), key=lambda u: (min(r.get("ts") or "" for r in by[u]), u))
    k = len(keep)
    cut = {"tune": keep[:k // 3], "held": keep[k // 3: 2 * k // 3], "conf": keep[2 * k // 3:]}
    out = {name: sorted((r for u in us for r in by[u]), key=lambda r: (r.get("ts") or "", r["id"])) for name, us in cut.items()}
    out["dropped_ids"] = sorted(r["id"] for u in bad for r in by[u])
    out["dropped_units"] = len(bad)
    out["units"] = cut
    return out


def structural_gates(rows: list[dict], meta: dict) -> dict:
    g = {"n": len(rows), "sessions": len({r.get("session") or r["id"] for r in rows}), "clusters": len({unit_of(r, meta) for r in rows}),
         "worlds": len({r.get("source") or "?" for r in rows})}
    g["pass"] = all(g[k] >= GATES[k] for k in ("n", "sessions", "clusters", "worlds"))
    g["failed"] = [k for k in ("n", "sessions", "clusters", "worlds") if g[k] < GATES[k]]
    return g


class RealBackend:
    """Qwen tokenizer counts, MiniLM vectors, the production Ollama model, the local judge. Loaded lazily."""

    def __init__(self, log=print):
        from transformers import AutoTokenizer
        from .frontend import ensure_base_in_ollama, local_model
        from .vectors import Encoder
        self.tok = AutoTokenizer.from_pretrained(config.base_path())
        self.enc = Encoder()
        ensure_base_in_ollama(log)
        self.model = local_model()
        self.name = {"local_model": self.model, "judge": config.JUDGE_MODEL}

    def ntok(self, s: str) -> int:
        return len(self.tok(s or "", add_special_tokens=False)["input_ids"])

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.enc.encode([t[:2000] for t in texts])

    def ask(self, instruction: str):
        from .frontend import ask_local
        return ask_local(self.model, instruction)

    def tool(self, vec) -> str | None:
        return None   # replayed agent work has no tool executor (R88: ~0 token value); the live path wires the exemplars

    def done_asking(self) -> None:
        from . import judge
        judge.unload(self.model)

    def grade(self, instruction: str, reference: str, answer: str) -> float:
        from . import judge
        return judge.grade(instruction, reference, answer)

    def close(self) -> None:
        from . import judge
        judge.unload()


def signals(recs_rows: list[dict], all_rows: list[dict], backend, log=print) -> list[dict]:
    """rolling-origin signals for each replayed row (bank = all rows with ts strictly earlier)"""
    allv = backend.encode([r["instruction"] for r in all_rows]) if all_rows else np.zeros((0, 384), np.float32)
    respv = backend.encode([chunk_text(r["response"]) for r in all_rows]) if all_rows else np.zeros((0, 384), np.float32)
    pos = {r["id"]: i for i, r in enumerate(all_rows)}
    ts = [parse_ts(r.get("ts")) for r in all_rows]
    chunk_tok: dict[int, int] = {}
    out = []
    for n, r in enumerate(recs_rows):
        i, t = pos[r["id"]], parse_ts(r.get("ts"))
        # strictly earlier (an exact-ts tie is NOT in the bank) and never a record of this same event: a same-session row within
        # SAME_EVENT_S may be the same exchange ingested twice and would carry the graded reference answer (cf. shadow_bank)
        bank_idx = [j for j in range(len(all_rows)) if t is not None and ts[j] is not None and ts[j] < t
                    and not (r.get("session") and all_rows[j].get("session") == r.get("session") and t - ts[j] <= SAME_EVENT_S)]
        bank = [all_rows[j] for j in bank_idx]
        hit, why = verified_cache_hit(r["instruction"], r.get("project"), r.get("ts"), bank)
        q = allv[i]
        max_sim = float((allv[bank_idx] @ q).max()) if bank_idx else None
        chunks = []
        if bank_idx:
            sc = respv[bank_idx] @ q
            for k in np.argsort(-sc)[:TOP_CHUNKS]:
                j = bank_idx[int(k)]
                if j not in chunk_tok:
                    chunk_tok[j] = backend.ntok(chunk_text(all_rows[j]["response"]))
                chunks.append({"id": all_rows[j]["id"], "score": float(sc[int(k)]), "tokens": chunk_tok[j]})
        if hit is not None:
            ans, conf, nt = hit, None, backend.ntok(hit)
        else:
            ans, conf, nt = backend.ask(r["instruction"])
        out.append({"id": r["id"], "ts": r.get("ts"), "session": r.get("session"), "source": r.get("source") or "?", "cache": hit is not None,
                    "cache_why": why, "tool": backend.tool(q), "max_sim": max_sim, "conf": conf, "ntok": nt, "tok_in": backend.ntok(r["instruction"]),
                    "tok_out": backend.ntok(r["response"]), "answer": ans, "chunks": chunks})
        if (n + 1) % 10 == 0:
            log(f"  signals {n + 1}/{len(recs_rows)}")
    return out


def decoy_reject(split_rows: list[dict], backend) -> tuple[float | None, int]:
    """judge calibration inside one split: Claude's answer to the nearest-but-different request must be rejected"""
    from .vectors import nearest_wrong
    if len(split_rows) < 2:
        return None, 0
    v = backend.encode([r["instruction"] for r in split_rows])
    scores = []
    for i, r in enumerate(split_rows):
        j = nearest_wrong(i, v, [x["response"] for x in split_rows])
        if j >= 0:
            scores.append(backend.grade(r["instruction"], r["response"], split_rows[j]["response"]))
    return (sum(1 for s in scores if s < WRONG) / len(scores) if scores else None), len(scores)


def tune(recs: list[dict]) -> tuple[dict | None, list[dict]]:
    grid = [{"min-conf": c, "min-sim": s} for c in GRID_CONF for s in GRID_SIM]
    swept = _sim(recs, grid, MAX_LOCAL_TOKENS)
    ok = [g for g in swept if g["wrong_rate"] <= TARGET_WRONG and g["local"] >= MIN_LOCAL_TUNE]
    best = max(ok, key=lambda g: (g["saved_tokens"], g["min-conf"], g["min-sim"])) if ok else None
    return best, swept


def fwd_score(r: dict) -> float:
    """the forward baseline of a row: the judge's score of Claude's OWN answer (reference graded against itself), i.e. what the user
    gets when the row is forwarded. q_delta = served score - this, so a judge that never gives 1.0 (ceiling bias) does not turn every
    correct served answer into a 'quality loss' (with score - 1.0, TOST +-0.01 was reachable only if every served answer scored 1.0).
    Missing baseline -> 1.0 (the ceiling: the conservative choice, any served shortfall counts in full)."""
    f = r.get("score_fwd")
    return float(f) if isinstance(f, (int, float)) else 1.0


def confirm(recs: list[dict], cfg: dict, decide) -> list[dict]:
    out = []
    for r in recs:
        d = decide(cfg, rec_signals(r), r["chunks"])
        served = d["decision"] != "forward"
        out.append({**r, "decision": d["decision"], "reason": d.get("reason"), "rust_ok": bool(d.get("ok")), "ctx_tokens": int(d.get("context_tokens") or 0),
                    "served": served, "wrong": served and r["score"] < WRONG,
                    "saved": (r["tok_in"] + r["tok_out"]) if served else -int(d.get("context_tokens") or 0),
                    "q_delta": (r["score"] - fwd_score(r)) if served else 0.0})
    return out


def _groups(rows: list[dict], key: str) -> list[list[float]]:
    g: dict[str, list[float]] = {}
    for r in rows:
        g.setdefault(r["unit"], []).append(float(r[key]))
    return list(g.values())


def verdict_tests(splits: dict[str, list[dict]], reps: int = REPS) -> dict:
    """one Holm family over every (split, world) x {saved>0, loss<0, TOST}; the KEEP rule"""
    tests = []
    for sp in ("held", "conf"):
        for w in sorted({r["source"] for r in splits[sp]}):
            rows = [r for r in splits[sp] if r["source"] == w]
            gs, gq = _groups(rows, "saved"), _groups(rows, "q_delta")
            tests.append({"split": sp, "world": w, "test": "saved_gt_0", "n": len(rows), "clusters": len(gs),
                          "mean": float(np.mean([r["saved"] for r in rows])), "p": stats.cluster_bootstrap_p(gs, "greater", 0.0, reps)})
            tests.append({"split": sp, "world": w, "test": "quality_loss_lt_0", "n": len(rows), "clusters": len(gq),
                          "mean": float(np.mean([r["q_delta"] for r in rows])), "p": stats.cluster_bootstrap_p(gq, "less", 0.0, reps)})
            t = stats.tost_cluster(gq, TOST_EPS, reps)
            tests.append({"split": sp, "world": w, "test": f"tost_quality_pm{TOST_EPS}", "n": len(rows), "clusters": len(gq), "mean": t["mean"], "p": t["p"]})
    for t, a in zip(tests, stats.holm([t["p"] for t in tests])):
        t["p_holm"] = a
        t["sig"] = a < ALPHA
    wrong = {}
    for sp in ("held", "conf"):
        served = [r for r in splits[sp] if r["served"]]
        k = sum(1 for r in served if r["wrong"])
        wrong[sp] = {"served": len(served), "wrong": k, "rate": (k / len(served)) if served else None, "ucb95": stats.wilson_upper(k, len(served))}
    sig = lambda sp, w, name: any(t["sig"] for t in tests if (t["split"], t["world"], t["test"]) == (sp, w, name))  # noqa: E731
    worlds_saved = sorted(w for w in {r["source"] for sp in ("held", "conf") for r in splits[sp]} if sig("held", w, "saved_gt_0") and sig("conf", w, "saved_gt_0"))
    loss = [t for t in tests if t["test"] == "quality_loss_lt_0" and t["sig"]]
    not_equiv = [t for t in tests if t["test"].startswith("tost_quality") and not t["sig"]]
    rust_err = sum(1 for sp in ("held", "conf") for r in splits[sp] if not r["rust_ok"])
    why = []
    if not worlds_saved:
        why.append("saved tokens not Holm-significant in one world on both HELD and CONF")
    if loss:
        why.append(f"Holm-significant quality loss in {[(t['split'], t['world']) for t in loss]}")
    if not_equiv:   # Iron rule: "no significant loss" is not enough, the quality delta must be shown EQUIVALENT within +-TOST_EPS
        why.append(f"quality delta not Holm-significantly equivalent within +-{TOST_EPS} (TOST) in {[(t['split'], t['world']) for t in not_equiv]}")
    for sp in ("held", "conf"):
        if wrong[sp]["ucb95"] > WRONG_UCB_MAX:
            why.append(f"{sp} wrong-rate 95% upper bound {wrong[sp]['ucb95']:.3f} > {WRONG_UCB_MAX}")
    if rust_err:
        why.append(f"{rust_err} Rust policy calls failed (forwarded)")
    return {"tests": tests, "wrong": wrong, "worlds_saved_both": worlds_saved, "rust_errors": rust_err,
            "verdict": "KEEP" if not why else "REJECTED", "why": why}


def _write(res: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "frontend_eval.json.tmp"
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    tmp.replace(out_dir / "frontend_eval.json")


def evaluate_v2(rows: list[dict], meta: dict, trained_ids, backend_factory, decide=rust_decide, out_dir: Path | None = None,
                log=print, reps: int = REPS, source: str = "") -> dict:
    """the whole v2 procedure; `backend_factory()` is only called when the structural gates pass"""
    out_dir = Path(out_dir or config.MODELS)
    sp = cluster_split(rows, meta, trained_ids)
    res = {"schema": SCHEMA, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data_source": source, "gates_required": GATES,
           "n_rows": len(rows), "dropped_trained_ids": len(sp["dropped_ids"]), "dropped_trained_units": sp["dropped_units"],
           "splits": {k: {"n": len(sp[k]), "units": len(sp["units"][k]), "first_ts": sp[k][0].get("ts") if sp[k] else None,
                          "last_ts": sp[k][-1].get("ts") if sp[k] else None} for k in ("tune", "held", "conf")},
           "token_unit": "Qwen tokenizer count of instruction+answer text (lower bound; excludes the ~95k agent prefix of R87)"}
    gates = {k: structural_gates(sp[k], meta) for k in ("tune", "held", "conf")}
    res["gates"] = gates
    if not all(g["pass"] for g in gates.values()):
        res.update(verdict="INSUFFICIENT", why=[f"{k}: {g['failed']}" for k, g in gates.items() if not g["pass"]])
        _write(res, out_dir)
        log(f"front-end eval v2: INSUFFICIENT ({res['why']}); no model was called")
        return res
    backend = backend_factory()
    res.update(getattr(backend, "name", {}))
    all_rows = sorted(rows, key=lambda r: (r.get("ts") or "", r["id"]))
    recs = {}
    for k in ("tune", "held", "conf"):
        recs[k] = signals(sp[k], all_rows, backend, log)
        for r in recs[k]:
            r["unit"] = unit_of(r, meta)
    backend.done_asking()
    for k in ("tune", "held", "conf"):
        for r, row in zip(recs[k], sp[k]):
            r["score"] = r["score_used"] = backend.grade(row["instruction"], row["response"], r["answer"])
            if k != "tune":   # forward baseline for q_delta (TUNE only uses the wrong rate)
                r["score_fwd"] = backend.grade(row["instruction"], row["response"], row["response"])
        gates[k]["decoy_reject"], gates[k]["n_decoys"] = decoy_reject(sp[k], backend)
        ok = gates[k]["decoy_reject"] is not None and gates[k]["decoy_reject"] >= GATES["decoy_reject"]
        gates[k]["pass"] = gates[k]["pass"] and ok
        if not ok:
            gates[k]["failed"].append("decoy_reject")
    backend.close()
    res["per_row"] = [{"split": k, **{f: r.get(f) for f in ("id", "unit", "source", "cache", "cache_why", "tool", "max_sim", "conf", "ntok", "tok_in", "tok_out", "score", "score_fwd")}}
                      for k in recs for r in recs[k]]
    if not all(g["pass"] for g in gates.values()):
        res.update(verdict="INSUFFICIENT", why=[f"{k}: {g['failed']}" for k, g in gates.items() if not g["pass"]])
        _write(res, out_dir)
        log(f"front-end eval v2: INSUFFICIENT ({res['why']})")
        return res
    best, swept = tune(recs["tune"])
    res["sweep_on_tune"] = swept
    if best is None:
        res.update(verdict="REJECTED", why=[f"no grid point on TUNE has wrong rate <= {TARGET_WRONG} with >= {MIN_LOCAL_TUNE} local answers"])
        _write(res, out_dir)
        log("front-end eval v2: REJECTED (no tuning point)")
        return res
    cfg = {"enabled": 1, "min-conf": best["min-conf"], "min-sim": best["min-sim"], "max-local-tokens": MAX_LOCAL_TOKENS, "ctx-budget": CTX_BUDGET}
    res["tuned"] = {"cfg": cfg, "tune_point": best}
    conf = {k: confirm(recs[k], cfg, decide) for k in ("held", "conf")}
    res["report"] = {k: {"n": len(v), **{d: sum(1 for r in v if r["decision"] == d) for d in ("cache", "tool", "local", "forward")},
                         "tokens_total": sum(r["tok_in"] + r["tok_out"] for r in v), "tokens_saved_net": sum(r["saved"] for r in v),
                         "forwarded_context_tokens_added": sum(r["ctx_tokens"] for r in v if not r["served"])} for k, v in conf.items()}
    res.update(verdict_tests(conf, reps))
    if res["verdict"] == "KEEP":
        cfg_out = {"schema": 1, "verdict": "KEEP", "eval_ts": res["ts"], **{k: v for k, v in cfg.items() if k != "enabled"},
                   "note": "thresholds only; the front-end is still enabled ONLY by an explicit --enabled (promotion ladder in llm/README.md)"}
        (out_dir / "frontend_cfg.json").write_text(json.dumps(cfg_out, indent=1), encoding="utf-8")
    _write(res, out_dir)
    log(f"front-end eval v2: {res['verdict']} {res['why']}")
    return res


def run(synthetic: bool = False, log=print) -> dict:
    if not binary_fresh():
        raise SystemExit("neural-mesh release binary is missing or older than src/frontend.rs / src/main.rs: run cargo build --release first")
    rows, source = data.load_rows(synthetic)
    meta = data.load_meta() if not synthetic else {}
    trained = config.load_state().get("trained_ids", []) if not synthetic else []
    return evaluate_v2(rows, meta, trained, lambda: RealBackend(log), out_dir=config.MODELS, log=log, source=source)

