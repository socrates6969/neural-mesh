"""`python -m llm eval`: candidate adapter vs the current production model on held-out LATER sessions.

Evidence, in the order the gate reads it:
(1) PAIRWISE judge (Claude Haiku via `claude -p`, llm/pairjudge.py): for each held-out request the candidate's and the
    production model's greedy answers are judged against Claude's real answer, in BOTH orders; a win needs both orders to
    agree, else tie. Score win=1 / tie=.5 / loss=0; 95% cluster-bootstrap CI over the ingestor's paraphrase clusters, plus an
    exact sign test on the decisive items. Hard per-night call budget (default 120), results cached by content hash.
    The judge must first have PASSED calibration (llm/calibrate.py: decoys >= 90%, deterministic agreement >= 85%); without
    that the gate stays closed.
(2) Deterministic battery (bench/battery.json, answers computed or unit-tested, $0): both models answer the same N items per
    category; no category may regress.
(3) held-out response-token NLL, paired, cluster-bootstrap CI: reported, and a significantly WORSE loss blocks promotion.
The old absolute local judge (qwen2.5-coder:7b) is no longer part of the gate (it accepted 40% of decoys); it remains only the
cheap pre-filter for ingest SFT rows.
Gate (promote.py applies it): calibrated judge AND >= 40 judged items AND win-rate CI lower bound > 0.5 AND no deterministic
category regression AND loss not significantly worse AND >= 3 clusters. Otherwise: not promoted.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import math
from datetime import datetime, timezone

from . import config, data, stats

N_LOSS = 200      # max held-out examples for the loss test
N_TASK = int(os.environ.get("MESH_PAIR_ITEMS", "1000"))  # max held-out requests judged pairwise (2 calls each); capped by the held-out pool
MIN_PAIR_ITEMS = 40   # judged held-out items needed before the win rate may open the gate
N_DET = 20        # deterministic battery items per category (math, physics, code, research)
DET_MAX_NEW = 768
CAT_MAX_DROP, POOLED_MAX_DROP, CAT_SIGN_ALPHA = 0.10, 0.05, 0.10
MIN_N, MIN_SESSIONS = 30, 3   # MIN_SESSIONS counts bootstrap units (ingestor clusters, else sessions)


def _pick(rows: list[dict], n: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256((salt + r["id"]).encode()).hexdigest())[:n]


def latest_candidate(synthetic: bool) -> str | None:
    if not config.ADAPTERS.exists():
        return None
    gone = {h.get("ver") for h in config.load_state().get("history", []) if h.get("action") == "abandoned"}
    vs = sorted(p.name for p in config.ADAPTERS.iterdir()
                if (p / "meta.json").exists() and p.name.endswith("-syn") == synthetic and p.name not in gone)
    return vs[-1] if vs else None


def _bench_module():
    spec = importlib.util.spec_from_file_location("bench_run", config.REPO / "bench" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det_items(n_per_cat: int) -> list[dict]:
    """first n_per_cat battery items per category (same stratification as bench/run.py --limit)."""
    seen: dict[str, int] = {}
    out = []
    for it in json.loads((config.REPO / "bench" / "battery.json").read_text(encoding="utf-8")):
        seen[it["cat"]] = seen.get(it["cat"], 0) + 1
        if seen[it["cat"]] <= n_per_cat:
            out.append(with_expected(it))
    return out


def det_correct(bench, item: dict, text: str) -> bool:
    """bench/run.py score() returns a dict (correct, fallback_correct, parsed, answer, sbx, infra); `bool(dict)` is always
    True, which made every battery answer count as correct. Use the strict `correct` flag. An infra failure (sandbox down)
    is neither right nor wrong: fail the eval closed instead of letting it skew the regression check."""
    if not text.strip():
        return False
    r = bench.score(item, text)
    if r.get("infra"):
        raise RuntimeError(f"battery scoring infra failure on {item['id']} (sandbox unavailable); eval aborted")
    return bool(r["correct"])


def with_expected(item: dict) -> dict:
    """code items need precomputed `expected` outputs since the host-side sandbox scoring (bench sandbox B1); the legacy
    battery.json has none. Compute them exactly as bench/battery.py tune_annotated does: run the trusted in-repo reference
    on each test, canonicalise with sandbox.canon, "EXC" when the reference raises. Non-code items and items that already
    carry `expected` are returned unchanged."""
    if item.get("kind") != "code" or "expected" in item:
        return item
    canon = _sandbox_module().canon
    g: dict = {}
    exec(item["ref"], g)  # noqa: S102 - reference solution shipped in bench/battery.json (repo-controlled, not model output)
    exp = []
    for t in item["tests"]:
        try:
            exp.append(canon(g[item["fn"]](*json.loads(json.dumps(t)))))
        except Exception:
            exp.append("EXC")
    return dict(item, expected=exp)


def _sandbox_module():
    spec = importlib.util.spec_from_file_location("bench_sandbox", config.REPO / "bench" / "sandbox.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def det_regressions(items: list[dict], ok_cand: list[bool], ok_prod: list[bool]) -> dict:
    """per-category paired outcomes. A category REGRESSES if the candidate's accuracy drops by more than CAT_MAX_DROP, or an
    exact one-sided sign test says it is worse at alpha CAT_SIGN_ALPHA; pooled over all items the drop must not exceed POOLED_MAX_DROP."""
    cats: dict[str, dict] = {}
    for it, c, p in zip(items, ok_cand, ok_prod):
        d = cats.setdefault(it["cat"], {"n": 0, "cand": 0, "prod": 0, "cand_only": 0, "prod_only": 0})
        d["n"] += 1
        d["cand"] += int(c)
        d["prod"] += int(p)
        d["cand_only"] += int(c and not p)
        d["prod_only"] += int(p and not c)
    bad = []
    for cat, d in cats.items():
        d["diff"] = (d["cand"] - d["prod"]) / d["n"]
        d["sign_p_worse"] = stats.sign_test_p(d["prod_only"], d["cand_only"])
        if d["diff"] < -CAT_MAX_DROP or d["sign_p_worse"] <= CAT_SIGN_ALPHA:
            bad.append(cat)
    pooled = (sum(ok_cand) - sum(ok_prod)) / max(1, len(items))
    if pooled < -POOLED_MAX_DROP:
        bad.append("pooled")
    return {"categories": cats, "pooled_diff": pooled, "n": len(items), "regressed": bad}


def gate(loss_ci: dict, pair: dict, det: dict, n_clusters: int, calibrated: tuple[bool, str] = (True, "")) -> tuple[bool, str]:
    """pair = {n_judged, winrate_ci{mean,lo,hi}}, det = det_regressions(...). Fail-closed on every missing piece."""
    if not calibrated[0]:
        return False, f"judge not trusted: {calibrated[1]}"
    if pair["n_judged"] < MIN_PAIR_ITEMS:
        return False, f"insufficient judged items (n={pair['n_judged']} < {MIN_PAIR_ITEMS}; budget, errors or too few held-out rows)"
    if n_clusters < MIN_SESSIONS:
        return False, f"too few clusters ({n_clusters} < {MIN_SESSIONS})"
    w = pair["winrate_ci"]
    if not w["lo"] > 0.5:
        return False, f"no significant pairwise win (win rate {w['mean']:.3f}, CI [{w['lo']:.3f}, {w['hi']:.3f}], lower bound must be > 0.5)"
    if det["regressed"]:
        return False, f"deterministic battery regression in: {', '.join(det['regressed'])}"
    if loss_ci["n"] and loss_ci["hi"] < 0:
        return False, f"held-out loss significantly worse (CI [{loss_ci['lo']:.4f}, {loss_ci['hi']:.4f}])"
    return True, f"pairwise win-rate CI lower bound {w['lo']:.3f} > 0.5 over {pair['n_judged']} items, no deterministic-category regression"


def pairwise_block(task_rows: list[dict], ans_c: list[str], ans_p: list[str], judge, unit, workers: int = 4) -> dict:
    """candidate (x) vs production (y) on the held-out requests; reference = the real recorded answer."""
    from . import pairjudge as PJ
    res = judge.compare_many([(r["instruction"], r["response"], a, b) for r, a, b in zip(task_rows, ans_c, ans_p)], workers)
    groups: dict = {}
    for r, x in zip(task_rows, res):
        sc = PJ.score_of(x)
        if sc is not None:
            groups.setdefault(unit(r), []).append(sc)
    s = PJ.summarize(res)
    return {"n_items": len(res), "n_judged": sum(len(g) for g in groups.values()), "clusters": len(groups),
            "winrate_ci": stats.cluster_bootstrap_ci(list(groups.values())),
            "wins": s["x_wins"], "losses": s["y_wins"], "ties": s["ties"], "unjudged": s["unjudged"],
            "sign_p_better": stats.sign_test_p(s["x_wins"], s["y_wins"]), "position_bias": s["position_bias"], "by_how": s["by_how"],
            "detail": [{"id": r["id"], "winner": x["winner"], "how": x["how"]} for r, x in zip(task_rows, res)]}


def _answers(cand, prod_ver, task_rows, items, m, tok, with_prod, log) -> dict:
    """greedy answers of both models; cached next to the adapter so a re-judge needs no GPU."""
    from . import model as M
    f = cand / "answers.json"
    key = {"prod": prod_ver, "task": [r["id"] for r in task_rows], "det": [i["id"] for i in items]}
    try:
        c = json.loads(f.read_text(encoding="utf-8"))
        if c.get("key") == key:
            log("reusing cached answers")
            return c["answers"]
    except (OSError, ValueError):
        pass
    instr = [r["instruction"] for r in task_rows]
    prompts = [i["prompt"] for i in items]
    out = {}
    m.set_adapter("cand")
    out["task_cand"] = M.generate_batch(m, tok, instr)
    out["det_cand"] = M.generate_batch(m, tok, prompts, max_new=DET_MAX_NEW)
    out["task_prod"] = with_prod(lambda: M.generate_batch(m, tok, instr))
    out["det_prod"] = with_prod(lambda: M.generate_batch(m, tok, prompts, max_new=DET_MAX_NEW))
    f.write_text(json.dumps({"key": key, "answers": out}), encoding="utf-8")
    return out


def run(synthetic: bool = False, ver: str | None = None, n_loss: int = N_LOSS, n_task: int = N_TASK, n_det: int = N_DET,
        judge_budget: int | None = None, runner=None, log=print) -> dict | None:
    import torch
    from peft import PeftModel
    from . import calibrate, judge, model as M, pairjudge as PJ
    explicit = ver is not None
    ver = ver or latest_candidate(synthetic)
    if not ver:
        log("no candidate adapter to evaluate")
        return None
    cand = config.ADAPTERS / ver
    if (cand / "eval.json").exists() and not explicit:
        log(f"{ver} already evaluated; nothing to do")
        return None
    meta = json.loads((cand / "meta.json").read_text(encoding="utf-8"))
    rows, source = data.load_rows(synthetic)
    _, held = data.split(rows, data.load_meta() if not synthetic else None)
    trained = set(meta["new_ids"]) | set(meta["replay_ids"])
    held = [r for r in held if r["id"] not in trained]   # paranoia: never score on trained rows
    st = config.load_state()
    prod = (st.get("synthetic_production") if synthetic else st.get("production")) or {}
    prod_ver = prod.get("ver")
    loss_rows = _pick(held, n_loss, "loss")
    task_rows = _pick(held, n_task, "task")
    if not synthetic and len(task_rows) < n_task:   # fill with the fixed open-source eval questions (never trained on)
        from . import external
        ext = [r for r in external.load_eval() if r["id"] not in trained]
        task_rows += _pick(ext, n_task - len(task_rows), "task")
        log(f"pairwise: {len(task_rows)} items (own held-out first, then fixed open-source eval questions)")
    items = det_items(n_det)
    log(f"eval {ver} vs production {prod_ver or 'BASE (no adapter)'}; held-out {len(held)} rows from {len({r.get('session') for r in held})} sessions; "
        f"loss n={len(loss_rows)}, pairwise n={len(task_rows)}, deterministic n={len(items)}")
    judge.unload()
    base = config.base_path(meta["base"])
    tok = M.tokenizer(base)
    m = M.load_4bit(base)
    m = PeftModel.from_pretrained(m, cand, adapter_name="cand")
    if prod_ver:
        m.load_adapter(config.ADAPTERS / prod_ver, adapter_name="prod")

    def with_prod(fn):
        if prod_ver:
            m.set_adapter("prod")
            return fn()
        with m.disable_adapter():
            return fn()

    enc = [M.encode(tok, r["instruction"], r["response"]) for r in loss_rows]
    m.set_adapter("cand")
    lc = M.example_losses(m, enc)
    lp = with_prod(lambda: M.example_losses(m, enc))
    smeta = data.load_meta() if not synthetic else {}
    unit = lambda r: (smeta.get(r["id"]) or {}).get("cluster") or r.get("session")  # noqa: E731  bootstrap unit: paraphrase cluster, else session
    per = [(unit(r), p[0] / max(1, p[1]) - c[0] / max(1, c[1])) for r, p, c in zip(loss_rows, lp, lc)]
    groups: dict = {}
    for s, d in per:
        groups.setdefault(s, []).append(d)
    loss_ci = stats.cluster_bootstrap_ci(list(groups.values()))
    loss_t = stats.paired_ci([d for _, d in per])
    nll_c = sum(c[0] for c in lc) / max(1, sum(c[1] for c in lc))
    nll_p = sum(p[0] for p in lp) / max(1, sum(p[1] for p in lp))
    log(f"held-out token NLL: production {nll_p:.4f} (ppl {math.exp(nll_p):.2f}) -> candidate {nll_c:.4f} (ppl {math.exp(nll_c):.2f}); "
        f"per-example diff {loss_ci['mean']:+.4f} cluster-CI [{loss_ci['lo']:+.4f}, {loss_ci['hi']:+.4f}]")
    ans = _answers(cand, prod_ver, task_rows, items, m, tok, with_prod, log)
    del m
    gc.collect()
    torch.cuda.empty_cache()
    bench = _bench_module()
    ok_c = [det_correct(bench, it, t) for it, t in zip(items, ans["det_cand"])]
    ok_p = [det_correct(bench, it, t) for it, t in zip(items, ans["det_prod"])]
    det = det_regressions(items, ok_c, ok_p)
    log("deterministic battery (cand vs prod correct/n): " + "; ".join(f"{c} {d['cand']}/{d['n']} vs {d['prod']}/{d['n']}" for c, d in det["categories"].items())
        + f"; pooled diff {det['pooled_diff']:+.3f}; regressed: {det['regressed'] or 'none'}")
    calibrated = calibrate.trusted(PJ.CLAUDE_MODEL)
    ledger = PJ.Ledger(kind="eval", cap=PJ.DEFAULT_NIGHTLY_BUDGET if judge_budget is None else judge_budget)
    if calibrated[0]:
        pj = PJ.PairwiseJudge(ledger, **({"runner": runner} if runner else {}))
        pair = pairwise_block(task_rows, ans["task_cand"], ans["task_prod"], pj, unit)
        pair.update(judge_stats=pj.stats, usage=pj.usage, ledger=ledger.snapshot(), resolved_model=pj.resolved_model)
    else:
        pair = {"n_items": len(task_rows), "n_judged": 0, "clusters": 0, "winrate_ci": stats.cluster_bootstrap_ci([]), "skipped": calibrated[1]}
    pw = pair["winrate_ci"]
    log(f"pairwise ({PJ.CLAUDE_MODEL}, both orders): judged {pair['n_judged']}/{pair['n_items']}, win rate {pw['mean']:.3f} CI [{pw['lo']:.3f}, {pw['hi']:.3f}]; "
        f"W/T/L {pair.get('wins')}/{pair.get('ties')}/{pair.get('losses')}; sign p {pair.get('sign_p_better')}; position bias {pair.get('position_bias')}; usage {pair.get('usage')}")
    ok, why = gate(loss_ci, pair, det, pair["clusters"], calibrated)
    res = {"ver": ver, "production": prod_ver, "data_source": source, "synthetic": synthetic, "base": meta["base"],
           "heldout_rows": len(held), "heldout_sessions": len(groups),
           "loss": {"nll_production": nll_p, "nll_candidate": nll_c, "ppl_production": math.exp(nll_p), "ppl_candidate": math.exp(nll_c),
                    "diff_cluster_ci": loss_ci, "diff_paired_t": loss_t},
           "pairwise": pair, "deterministic": det, "judge_trusted": {"ok": calibrated[0], "why": calibrated[1]},
           "gate": {"promote": ok, "reason": why}, "quality": quality_report(det, pair), "evaluated": datetime.now(timezone.utc).isoformat()}
    (cand / "eval.json").write_text(json.dumps(res, indent=1, default=lambda x: None if isinstance(x, float) and math.isinf(x) else x), encoding="utf-8")
    log(describe_quality(res["quality"]))
    log(f"gate: {'PROMOTE' if ok else 'KEEP PRODUCTION'} - {why}")
    return res

# ---- E2: baseline-aware, powered reporting (does not change the gate decision) ---------------------------------------------------------
PLANNING_DISCORDANCE = 0.30   # assumed share of discordant items when planning an MDE for a comparison that has none yet (a labelled assumption)


def _cat_row(d: dict) -> dict:
    n, b, c = d["n"], d["cand_only"], d["prod_only"]
    return {"n": n, "recall_cand": d["cand"] / n, "recall_prod": d["prod"] / n, "cand_only": b, "prod_only": c,
            "mcnemar_p_two_sided": stats.mcnemar_exact_p(b, c), "sign_p_cand_better": stats.sign_test_p(b, c),
            "mde_at_observed": stats.mde_observed(n, b, c), "mde_planning": stats.mde_paired(n, PLANNING_DISCORDANCE)}


def quality_report(det: dict, pair: dict, target_win_share: float = 0.6) -> dict:
    """What the stored counts can and cannot support. det = det_regressions(...) output, pair = pairwise_block(...) output.
    Baselines: NO-OP (candidate answers == production answers: every pair a tie, win rate .5, accuracy difference 0 by construction) and
    EMPTY ANSWER (0 correct). Balanced accuracy = macro mean of the per-category recalls (one vote per category, whatever its size)."""
    cats = {k: _cat_row(d) for k, d in det["categories"].items()}
    nn = sum(d["n"] for d in det["categories"].values())
    b = sum(d["cand_only"] for d in det["categories"].values())
    c = sum(d["prod_only"] for d in det["categories"].values())
    kc, kp = sum(d["cand"] for d in det["categories"].values()), sum(d["prod"] for d in det["categories"].values())
    bal_c = sum(r["recall_cand"] for r in cats.values()) / len(cats) if cats else None
    bal_p = sum(r["recall_prod"] for r in cats.values()) / len(cats) if cats else None
    pooled = {"n": nn, "acc_cand": kc / nn if nn else None, "acc_prod": kp / nn if nn else None, "cand_only": b, "prod_only": c,
              "balanced_acc_cand": bal_c, "balanced_acc_prod": bal_p, "mcnemar_p_two_sided": stats.mcnemar_exact_p(b, c),
              "mde_at_observed": stats.mde_observed(nn, b, c) if nn else None, "mde_planning": stats.mde_paired(nn, PLANNING_DISCORDANCE) if nn else None}
    w, l, t = pair.get("wins") or 0, pair.get("losses") or 0, pair.get("ties") or 0
    m = w + l
    need = stats.required_decisive(target_win_share)
    pw = {"wins": w, "losses": l, "ties": t, "judged": pair.get("n_judged", 0), "items": pair.get("n_items", 0), "decisive": m,
          "win_share_among_decisive": w / m if m else None, "mcnemar_p_two_sided": stats.mcnemar_exact_p(w, l),
          "sign_p_cand_better": stats.sign_test_p(w, l), "noop_win_rate": 0.5, "mde_win_share": stats.mde_sign_test(m),
          "target_win_share": target_win_share, "decisive_needed_for_target": need, "powered_for_target": m >= need if need else False}
    pc = pooled["mde_at_observed"]
    pooled_under = pc is None or pc["min_diff"] is None or pc["min_diff"] > 0.10
    return {"categories": cats, "pooled": pooled, "pairwise": pw, "baselines": {"noop_acc_diff": 0.0, "empty_answer_acc": 0.0},
            "underpowered": {"battery_for_10pt": pooled_under, "pairwise_for_target": not pw["powered_for_target"]}}


def describe_quality(q: dict) -> str:
    p, pw = q["pooled"], q["pairwise"]
    f = lambda x: "n/a" if x is None else f"{x:.3f}"   # noqa: E731
    mde = (p["mde_at_observed"] or {}).get("min_diff")
    return (f"quality: battery pooled {f(p['acc_cand'])} vs {f(p['acc_prod'])}, balanced {f(p['balanced_acc_cand'])} vs {f(p['balanced_acc_prod'])}, "
            f"McNemar p {p['mcnemar_p_two_sided']:.3f}, MDE(80% power) {f(mde)} acc points (underpowered for 10 pts: {q['underpowered']['battery_for_10pt']}); "
            f"pairwise decisive {pw['decisive']} (W {pw['wins']}/L {pw['losses']}/T {pw['ties']}), MDE win share {f(pw['mde_win_share'])}, "
            f"{pw['decisive_needed_for_target']} decisive pairs needed for a {pw['target_win_share']:.0%} win share")


def rescore(path) -> dict:
    """re-score a stored eval.json from its stored counts (read-only)."""
    from pathlib import Path
    e = json.loads(Path(path).read_text(encoding="utf-8"))
    q = quality_report(e["deterministic"], e["pairwise"])
    return {"ver": e.get("ver"), "stored_gate": e.get("gate"), "stored_regressed": e["deterministic"].get("regressed"), "quality": q}


if __name__ == "__main__":
    import sys
    r = rescore(sys.argv[1])
    print(json.dumps(r, indent=1))
    print(describe_quality(r["quality"]))
