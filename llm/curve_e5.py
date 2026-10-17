"""E5 learning curve on the NEW verified rows (exp-data, hive experiment; scratch only).

`python -m llm.curve_e5 split|train --frac F|eval|pairwise` with MESH_LLM_DATA = an ingest data dir that has the exec-verified
tooluse_sft.jsonl + sft.jsonl, and MESH_LLM_MODELS = a SCRATCH models dir (adapters land in <models>/adapters/e5-<pct>; never
Ollama, never state.json). Primary graded metric: next-tool accuracy on held-out (later) tool-use steps, parsed from the generated
"ACTION: Tool[...]" line; baselines BASE and the majority tool. Secondary: Haiku pairwise on held-out answer rows.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from . import config

FORMAT = "\n\nReply with a short reason, then one line: ACTION: Tool[command verbs or file extension](argument keys)"
STEPS_PER_TRAJ = 12
EVAL_PER_TRAJ = 10
EVAL_MAX = 300
HELD_FRAC = 0.2
FRACS = (25, 50, 100)
_ACTION = re.compile(r"ACTION:\s*([A-Za-z0-9_.:-]+)\[([^\]]*)\]")


def _rows(p: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    except FileNotFoundError:
        return []


def split_path() -> Path:
    return config.MODELS / "e5_split.json"


def make_split(seed: int = config.SEED) -> dict:
    steps = [s for s in _rows(config.DATA / "tooluse_sft.jsonl") if s.get("verified") is True]
    by_t: dict[str, list[dict]] = {}
    for s in steps:
        by_t.setdefault(s["traj"], []).append(s)
    trajs = sorted(by_t, key=lambda t: (min(x["ts"] or "" for x in by_t[t]), t))
    cut = int(round(len(trajs) * (1 - HELD_FRAC)))
    train_t, held_t = trajs[:cut], trajs[cut:]
    rng = random.Random(seed)
    order = train_t[:]
    rng.shuffle(order)
    keep = {}
    for t in trajs:
        xs = sorted(by_t[t], key=lambda x: x["step"])
        k = STEPS_PER_TRAJ if t in set(train_t) else EVAL_PER_TRAJ
        keep[t] = [x["id"] for x in (xs if len(xs) <= k else random.Random(seed + len(t)).sample(xs, k))]
    held_steps = [i for t in held_t for i in keep[t]][:EVAL_MAX]
    ans = sorted(_rows(config.DATA / "sft.jsonl"), key=lambda r: (r.get("ts") or "", r["id"]))
    acut = int(round(len(ans) * (1 - HELD_FRAC)))
    ans_train, ans_held = [r["id"] for r in ans[:acut]], [r["id"] for r in ans[acut:]]
    rng.shuffle(ans_train)
    sets = {}
    for f in FRACS:
        nt, na = max(1, round(len(order) * f / 100)), max(1, round(len(ans_train) * f / 100))
        sets[str(f)] = {"trajs": order[:nt], "steps": [i for t in order[:nt] for i in keep[t]], "answers": ans_train[:na]}
    tool_counts = Counter(s["action"]["n"] for t in train_t for s in by_t[t])
    out = {"seed": seed, "n_traj": len(trajs), "train_trajs": len(train_t), "held_trajs": len(held_t), "held_steps": held_steps,
           "held_answers": ans_held, "sets": sets, "majority_tool": tool_counts.most_common(1)[0][0],
           "train_tool_counts": dict(tool_counts.most_common())}
    config.MODELS.mkdir(parents=True, exist_ok=True)
    split_path().write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def _load_split() -> dict:
    return json.loads(split_path().read_text(encoding="utf-8"))


def _index() -> dict[str, dict]:
    idx = {s["id"]: s for s in _rows(config.DATA / "tooluse_sft.jsonl")}
    idx.update({r["id"]: r for r in _rows(config.DATA / "sft.jsonl")})
    return idx


def _instr(r: dict) -> str:
    return r["instruction"] + (FORMAT if r.get("source") == "tool_step" else "")


def train(frac: int, epochs: int = 2, log=print) -> dict:
    from . import judge, model as M
    sp, idx = _load_split(), _index()
    s = sp["sets"][str(frac)]
    rows = [idx[i] for i in s["steps"] + s["answers"]]
    judge.unload()
    base = config.base_path()
    tok = M.tokenizer(base)
    enc = [M.encode(tok, _instr(r), r["response"]) for r in rows]
    log(f"e5 train {frac}%: {len(s['steps'])} steps from {len(s['trajs'])} trajectories + {len(s['answers'])} answer rows; "
        f"mean {sum(len(e[0]) for e in enc) / len(enc):.0f} tokens")
    m = M.load_4bit(base, trainable=True)
    stats = M.train_loop(m, enc, epochs=epochs, log=log)
    out = config.ADAPTERS / f"e5-{frac:03d}"
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    meta = {"frac": frac, "n_steps": len(s["steps"]), "n_trajs": len(s["trajs"]), "n_answers": len(s["answers"]), "epochs": epochs,
            "train": stats, "base": base.name}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"adapter {out}: {stats}")
    return meta


def parse_action(text: str) -> tuple[str | None, str | None]:
    hits = _ACTION.findall(text or "")
    if not hits:
        return None, None
    tool, tgt = hits[-1]
    first = tgt.split(",")[0].strip().lower() or None
    return tool, first


def _gold(r: dict) -> tuple[str, str | None]:
    a = r["action"]
    first = (a.get("v") or [None])[0] or a.get("x")
    return a["n"], (first.lower() if first else None)


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def _battery(m, tok, M, E, bench, items, log) -> dict:
    """deterministic battery (math/physics/code/research, 20 each): per-category correct flags; an item whose scoring needs an
    unavailable sandbox is recorded as infra and excluded (never counted right or wrong)."""
    outs = M.generate_batch(m, tok, [it["prompt"] for it in items], max_new=E.DET_MAX_NEW)
    flags = []
    for it, t in zip(items, outs):
        try:
            flags.append(E.det_correct(bench, it, t))
        except RuntimeError:
            flags.append(None)
    return {"flags": flags}


def evaluate(max_new: int = 160, n_det: int = 20, log=print) -> dict:
    from peft import PeftModel
    from . import evaluate as E, judge, model as M
    sp, idx = _load_split(), _index()
    held = [idx[i] for i in sp["held_steps"]]
    items = E.det_items(n_det) if n_det else []
    bench = E._bench_module() if items else None
    kinds = [r["action"].get("k") or "other" for r in held]
    held_ans = [idx[i] for i in sp["held_answers"]]
    judge.unload()
    base = config.base_path()
    tok = M.tokenizer(base)
    m = M.load_4bit(base)
    vers = [f"e5-{f:03d}" for f in FRACS if (config.ADAPTERS / f"e5-{f:03d}" / "meta.json").exists()]
    for i, v in enumerate(vers):
        if i == 0:
            m = PeftModel.from_pretrained(m, config.ADAPTERS / v, adapter_name=v)
        else:
            m.load_adapter(config.ADAPTERS / v, adapter_name=v)
    prompts = [_instr(r) for r in held]
    golds = [_gold(r) for r in held]
    res: dict = {"held_steps": len(held), "held_trajs": len({r["traj"] for r in held}), "held_answers": len(held_ans),
                 "majority_tool": sp["majority_tool"], "models": {}}
    maj = [g[0] == sp["majority_tool"] for g in golds]
    res["models"]["MAJORITY"] = {"tool_acc": sum(maj) / len(maj), "tool_ci": wilson(sum(maj), len(maj))}
    answers = {}

    def run(name: str):
        outs = M.generate_batch(m, tok, prompts, max_new=max_new)
        ans = M.generate_batch(m, tok, [r["instruction"] for r in held_ans], max_new=384) if held_ans else []
        parsed = [parse_action(o) for o in outs]
        tool_ok = [p[0] == g[0] for p, g in zip(parsed, golds)]
        full_ok = [p[0] == g[0] and (g[1] is None or p[1] == g[1]) for p, g in zip(parsed, golds)]
        bat = _battery(m, tok, M, E, bench, items, log) if items else {"flags": []}
        answers[name] = {"steps": outs, "answers": ans, "tool_ok": tool_ok, "battery": bat["flags"]}
        k = sum(tool_ok)
        per_kind = {}
        for kd, ok in zip(kinds, tool_ok):
            c = per_kind.setdefault(kd, [0, 0]); c[0] += int(ok); c[1] += 1
        per_cat = {}
        for it, ok in zip(items, bat["flags"]):
            c = per_cat.setdefault(it["cat"], [0, 0, 0]); c[2] += int(ok is None)
            if ok is not None:
                c[0] += int(ok); c[1] += 1
        res["models"][name] = {"tool_acc": k / len(held), "tool_ci": wilson(k, len(held)), "full_acc": sum(full_ok) / len(held),
                               "full_ci": wilson(sum(full_ok), len(held)), "parsed": sum(1 for p in parsed if p[0]) / len(held),
                               "tools_predicted": dict(Counter(p[0] for p in parsed).most_common(8)),
                               "tool_acc_by_kind": {kd: {"correct": a, "n": n} for kd, (a, n) in sorted(per_kind.items())},
                               "battery_by_cat": {c: {"correct": a, "n": n, "infra": x} for c, (a, n, x) in sorted(per_cat.items())}}
        log(f"  {name}: tool acc {k}/{len(held)} = {k / len(held):.3f} {res['models'][name]['tool_ci']}, full {sum(full_ok)}, "
            f"parsed {res['models'][name]['parsed']:.2f}")

    if vers:
        with m.disable_adapter():
            run("BASE")
        for v in vers:
            m.set_adapter(v)
            run(v)
    else:
        run("BASE")

    def paired(a: str, b: str) -> dict:
        x, y = answers[a]["tool_ok"], answers[b]["tool_ok"]
        bb = sum(1 for i, j in zip(x, y) if i and not j)
        cc = sum(1 for i, j in zip(x, y) if j and not i)
        return {"a_only": bb, "b_only": cc, "mcnemar_p": mcnemar_p(bb, cc)}
    res["paired"] = {f"{v} vs BASE": paired(v, "BASE") for v in vers}
    if items:
        cats = sorted({it["cat"] for it in items})
        res["battery_paired_vs_base"] = {}
        for v in vers:
            for c in cats:
                ix = [i for i, it in enumerate(items) if it["cat"] == c]
                x = [answers[v]["battery"][i] for i in ix]
                y = [answers["BASE"]["battery"][i] for i in ix]
                b = sum(1 for i, j in zip(x, y) if i is True and j is False)
                cc = sum(1 for i, j in zip(x, y) if j is True and i is False)
                res["battery_paired_vs_base"][f"{v}:{c}"] = {"adapter_only": b, "base_only": cc, "mcnemar_p": mcnemar_p(b, cc)}
    if "e5-100" in answers and "e5-025" in answers:
        res["paired"]["e5-100 vs e5-025"] = paired("e5-100", "e5-025")
    mj = {"tool_ok": maj}
    answers["MAJORITY"] = mj
    for v in vers:
        res["paired"][f"{v} vs MAJORITY"] = paired(v, "MAJORITY")
    (config.MODELS / "e5_answers.json").write_text(json.dumps({"held_steps": sp["held_steps"], "held_answers": sp["held_answers"],
                                                               "answers": answers}, indent=0), encoding="utf-8")
    (config.MODELS / "e5_eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def pairwise(vers: list[str], log=print) -> dict:
    from . import pairjudge as PJ
    from . import stats
    a = json.loads((config.MODELS / "e5_answers.json").read_text(encoding="utf-8"))
    idx = _index()
    held = [idx[i] for i in a["held_answers"]]
    ledger = PJ.Ledger(config.MODELS / "pair_budget.json", kind="e5", cap=4 * len(held) * len(vers) + 8)
    pj = PJ.PairwiseJudge(ledger, cache_path=config.MODELS / "pair_cache.jsonl")
    out = {}
    for v in vers:
        items = [(r["instruction"], r["response"], x, y) for r, x, y in zip(held, a["answers"][v]["answers"], a["answers"]["BASE"]["answers"])]
        res = pj.compare_many(items, workers=4)
        s = PJ.summarize(res)
        sc = [PJ.score_of(x) for x in res]
        sc = [x for x in sc if x is not None]
        out[v] = {"n": len(held), "judged": len(sc), "wins": s["x_wins"], "losses": s["y_wins"], "ties": s["ties"],
                  "winrate": (sum(sc) / len(sc)) if sc else None, "ci": stats.cluster_bootstrap_ci([[x] for x in sc]) if sc else None,
                  "sign_p": stats.sign_test_p(s["x_wins"], s["y_wins"])}
        log(f"  {v} vs BASE: {out[v]}")
    (config.MODELS / "e5_pairwise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m llm.curve_e5")
    p.add_argument("cmd", choices=["split", "train", "eval", "pairwise"])
    p.add_argument("--frac", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--vers", default="e5-100")
    a = p.parse_args(argv)
    if a.cmd == "split":
        s = make_split()
        print(json.dumps({k: (v if not isinstance(v, (list, dict)) else len(v)) for k, v in s.items()}))
    elif a.cmd == "train":
        train(a.frac, a.epochs)
    elif a.cmd == "eval":
        print(json.dumps(evaluate(), indent=1))
    else:
        print(json.dumps(pairwise(a.vers.split(",")), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())