"""Front-end (feature 1, opt-in, DEFAULT OFF): answer locally or forward to Claude with retrieved, trimmed context.

Signals per request: exact cache (normalised instruction seen in the trained set), OOD similarity (max MiniLM-384 cosine
to the trained instructions), local answer confidence (mean token log-prob from Ollama `logprobs`) and length. The
DECISION is made by the Rust policy (`neural-mesh frontend`, src/frontend.rs) so the mesh owns it; retrieved context is
ranked by cosine and trimmed by the same binary.

`python -m llm frontend-eval` replays held-out (later-session) real sft rows: Claude's actual answer is the reference,
the local judge grades the local answer (0/0.5/1), thresholds are TUNED on the earlier half of the replay sessions and
REPORTED on the later half. Token numbers are Qwen-tokenizer counts of instruction+answer text: a LOWER BOUND on the real
cost of a forwarded turn (R87: a delegated agent run carries a ~95k-token prefix per API call, which a local answer
would also avoid, but that is not attributed here)."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

from . import config, data, judge

BIN = config.REPO / "target" / "release" / ("neural-mesh.exe" if (config.REPO / "target" / "release" / "neural-mesh.exe").exists() else "neural-mesh")
WRONG = 0.5          # judge score below this = a wrong local answer
TARGET_WRONG = 0.10  # tuning rule: maximise local share s.t. wrong-local rate <= 10% on the TUNE half


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def rust_decide(cfg: dict, sig: dict, chunks: list[dict]) -> dict:
    args = [str(BIN), "frontend"] + [x for k, v in cfg.items() for x in (f"--{k}", str(v))]
    for k, v in sig.items():
        if v is not None:
            args += [f"--{k}", str(v)]
    if chunks:
        args += ["--chunks", json.dumps(chunks)]
    r = subprocess.run(args, capture_output=True, text=True)
    return json.loads(r.stdout)


def local_model() -> str:
    st = config.load_state()
    if (st.get("production") or {}).get("ollama"):
        return f"{config.OLLAMA_NAME}:latest"
    return f"{config.OLLAMA_NAME}:base"


def ensure_base_in_ollama(log=print) -> None:
    from .promote import ollama_import
    import urllib.request
    with urllib.request.urlopen(config.OLLAMA_URL + "/api/tags", timeout=30) as r:
        names = {m["name"] for m in json.loads(r.read())["models"]}
    if f"{config.OLLAMA_NAME}:base" not in names:
        ollama_import(config.base_path(), f"{config.OLLAMA_NAME}:base", log=log)


def ask_local(model: str, instruction: str, max_new: int = 400) -> tuple[str, float, int]:
    r = judge._post("/api/chat", {"model": model, "stream": False, "logprobs": True,
                                  "messages": [{"role": "user", "content": instruction[-12000:]}],
                                  "options": {"temperature": 0, "seed": 0, "num_predict": max_new, "num_ctx": 8192}})
    lps = [x["logprob"] for x in (r.get("logprobs") or [])]
    return r.get("message", {}).get("content", "").strip(), (sum(lps) / len(lps) if lps else -99.0), int(r.get("eval_count", len(lps)))


def _sim(dec_rows, cfgs):
    out = []
    for c in cfgs:
        loc = [r for r in dec_rows if r["cache"] or (r["max_sim"] >= c["min-sim"] and r["conf"] >= c["min-conf"] and 0 < r["ntok"] <= 400)]
        wrong = sum(1 for r in loc if r["score_used"] < WRONG)
        out.append({**c, "local": len(loc), "wrong": wrong, "wrong_rate": wrong / len(loc) if loc else 0.0,
                    "saved_tokens": sum(r["tok_in"] + r["tok_out"] for r in loc), "total_tokens": sum(r["tok_in"] + r["tok_out"] for r in dec_rows)})
    return out


def evaluate(synthetic: bool = False, n: int = 80, log=print) -> dict:
    from transformers import AutoTokenizer
    from .vectors import Encoder, max_cos, nearest_wrong
    rows, source = data.load_rows(synthetic)
    train, held = data.split(rows, data.load_meta() if not synthetic else None)
    held = sorted(held, key=lambda r: r.get("ts") or "")
    if len(held) > n:  # even spread over time, keeps session order
        idx = np.linspace(0, len(held) - 1, n).round().astype(int)
        held = [held[i] for i in sorted(set(idx))]
    tok = AutoTokenizer.from_pretrained(config.base_path())
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])  # noqa: E731
    enc = Encoder()
    bank_v = enc.encode([r["instruction"][:2000] for r in train])
    held_v = enc.encode([r["instruction"][:2000] for r in held])
    resp_v = enc.encode([r["response"][:2000] for r in train]) if train else np.zeros((0, 384), np.float32)
    cache = {norm(r["instruction"]): r["response"] for r in train}
    ensure_base_in_ollama(log)
    model = local_model()
    log(f"front-end replay: {source}; bank {len(train)} trained rows, replay {len(held)} later-session rows; local model {model}; judge {config.JUDGE_MODEL}")
    recs = []
    sims = max_cos(held_v, bank_v)
    for i, r in enumerate(held):
        hit = cache.get(norm(r["instruction"]))
        ans, conf, nt = (hit, 0.0, ntok(hit)) if hit else ask_local(model, r["instruction"])
        # retrieval context for a forward: top bank answers by cosine to the request
        sc = (resp_v @ held_v[i]) if len(resp_v) else np.zeros(0)
        top = np.argsort(-sc)[:8]
        chunks = [{"id": train[j]["id"], "score": float(sc[j]), "tokens": ntok(train[j]["response"][:4000])} for j in top]
        recs.append({"id": r["id"], "session": r.get("session"), "source": r.get("source"), "cache": bool(hit), "max_sim": float(sims[i]),
                     "conf": conf, "ntok": nt, "tok_in": ntok(r["instruction"]), "tok_out": ntok(r["response"]), "answer": ans, "chunks": chunks})
        if (i + 1) % 10 == 0:
            log(f"  local answers {i + 1}/{len(held)}")
    judge.unload(model)
    for r, h in zip(recs, held):
        r["score"] = judge.grade(h["instruction"], h["response"], r["answer"])
    # judge calibration with hard decoys: Claude's answer to the NEAREST-BUT-DIFFERENT held request should be rejected
    hv = held_v
    decoy = []
    for i, h in enumerate(held):
        j = nearest_wrong(i, hv, [x["response"] for x in held])
        if j >= 0:
            decoy.append(judge.grade(h["instruction"], h["response"], held[j]["response"]))
    judge.unload()
    decoy_reject = sum(1 for s in decoy if s < WRONG) / len(decoy) if decoy else None
    for r in recs:
        r["score_used"] = r["score"]
    # tune on the EARLIER half of the replay (by time), report on the LATER half (sessions can be few/one)
    half = len(recs) // 2                       # recs follow `held`, which is sorted by ts
    tune, test = recs[:half], recs[half:]
    sessions = sorted({r["session"] for r in recs})
    grid = [{"min-conf": c, "min-sim": s} for c in (-0.02, -0.05, -0.1, -0.15, -0.2, -0.3, -0.5, -1.0, -99.0) for s in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7)]
    tuned = _sim(tune, grid)
    ok = [g for g in tuned if g["wrong_rate"] <= TARGET_WRONG]
    best = max(ok, key=lambda g: (g["saved_tokens"], g["min-conf"], g["min-sim"])) if ok else {"min-conf": 99.0, "min-sim": 1.0}
    chosen = {"enabled": 1, "min-conf": best["min-conf"], "min-sim": best["min-sim"], "max-local-tokens": 400, "ctx-budget": 800}
    # the REPORT half goes through the real Rust policy (the hook), not the python simulation
    rep = []
    for r in test:
        d = rust_decide(chosen, {"cache-hit": int(r["cache"]), "local-conf": r["conf"], "local-tokens": r["ntok"], "max-sim": r["max_sim"]}, r["chunks"])
        rep.append({**r, "decision": d["decision"], "ctx_tokens": d["context_tokens"]})
    loc = [r for r in rep if r["decision"] != "forward"]
    fwd = [r for r in rep if r["decision"] == "forward"]
    tot = sum(r["tok_in"] + r["tok_out"] for r in rep)
    res = {
        "data_source": source, "local_model": model, "judge": config.JUDGE_MODEL, "n_replay": len(recs), "sessions": len(sessions),
        "token_unit": "Qwen tokenizer count of instruction+answer text (lower bound; excludes the ~95k agent prefix of R87)",
        "judge_decoy_reject_rate": decoy_reject, "n_decoys": len(decoy),
        "local_answer_quality_all": {"mean_score": float(np.mean([r["score"] for r in recs])) if recs else None,
                                     "wrong_rate_if_always_local": sum(r["score"] < WRONG for r in recs) / max(1, len(recs))},
        "tuned_on_earlier_half": {"n": len(tune), "rule": f"max saved tokens s.t. wrong-local <= {TARGET_WRONG}", "chosen": chosen, "tune_point": best if ok else None},
        "report_on_later_half": {"n": len(rep), "local": len(loc), "forward": len(fwd), "local_share": len(loc) / max(1, len(rep)),
                                 "wrong_local": sum(r["score"] < WRONG for r in loc), "wrong_local_rate": (sum(r["score"] < WRONG for r in loc) / len(loc)) if loc else None,
                                 "tokens_total": tot, "tokens_saved": sum(r["tok_in"] + r["tok_out"] for r in loc),
                                 "saved_share": sum(r["tok_in"] + r["tok_out"] for r in loc) / max(1, tot),
                                 "forwarded_context_tokens_added": sum(r["ctx_tokens"] for r in fwd)},
        "sweep_on_tune_half": tuned,
        "per_row": [{k: r[k] for k in ("id", "source", "cache", "max_sim", "conf", "ntok", "tok_in", "tok_out", "score")} for r in recs],
    }
    out = config.MODELS / "frontend_eval.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    rp = res["report_on_later_half"]
    log(f"decoy reject rate {decoy_reject}; always-local wrong rate {res['local_answer_quality_all']['wrong_rate_if_always_local']:.2f}; "
        f"chosen {chosen}; later half: local {rp['local']}/{rp['n']}, wrong-local {rp['wrong_local']}, tokens saved {rp['tokens_saved']}/{rp['tokens_total']} ({rp['saved_share']:.1%})")
    return res
