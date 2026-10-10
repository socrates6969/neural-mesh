"""Measured base-model choice: QLoRA step time + peak VRAM (seq 512/1024, bs 1) and greedy decode speed per candidate.
Writes models/bench.json and models/base_choice.json (largest candidate whose 1024-token step fits under the VRAM budget
with headroom, ties -> faster)."""
from __future__ import annotations

import gc
import json
import time

from . import config

CANDIDATES = ["Qwen2.5-Coder-1.5B-Instruct", "Qwen2.5-Coder-3B-Instruct"]
VRAM_BUDGET_GB = 6.0   # of 8 GB: leave room for the desktop and a resident Ollama model


def run(log=print) -> dict:
    import torch
    from . import model as M
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda, "candidates": {}}
    for name in CANDIDATES:
        base = config.base_path(name)
        if not base.exists():
            log(f"skip {name}: not downloaded"); continue
        tok = M.tokenizer(base)
        r = {}
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t = time.time(); m = M.load_4bit(base, trainable=True); r["load_s"] = round(time.time() - t, 1)
        r["load_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        for L in (512, 1024):
            ids = [list(range(1000, 1000 + L))]
            enc = [(ids[0], ids[0][:])] * 12
            out = M.train_loop(m, enc, epochs=1, bs=1, grad_accum=4, log=lambda *_: None)
            # first optimizer step includes warm-up/allocation: time the steady state with a second pass
            torch.cuda.reset_peak_memory_stats()
            t = time.time(); M.train_loop(m, enc, epochs=1, bs=1, grad_accum=4, log=lambda *_: None)
            dt = time.time() - t
            r[f"seq{L}"] = {"s_per_example": round(dt / len(enc), 3), "tokens_per_s": round(L * len(enc) / dt), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            log(f"{name} seq{L}: {r[f'seq{L}']}")
        txt, lp, n = M.generate(m, tok, "Write a Python function that checks whether a string is a palindrome.", max_new=96)
        t = time.time(); txt, lp, n = M.generate(m, tok, "Explain in two sentences what a mutex is.", max_new=96); dt = time.time() - t
        r["decode_tok_per_s_hf4bit"] = round(n / dt, 1)
        log(f"{name} decode {r['decode_tok_per_s_hf4bit']} tok/s (HF 4-bit, greedy, n={n})")
        res["candidates"][name] = r
        del m; gc.collect(); torch.cuda.empty_cache()
    ok = [(n, c) for n, c in res["candidates"].items() if c["seq1024"]["peak_vram_gb"] <= VRAM_BUDGET_GB]
    # prefer the bigger model if it fits and its 1024-token step is <= 3x the small one's (nightly budget), else the small one
    choice = config.DEFAULT_BASE
    if ok:
        small = res["candidates"].get(CANDIDATES[0])
        big = res["candidates"].get(CANDIDATES[1])
        if big and big["seq1024"]["peak_vram_gb"] <= VRAM_BUDGET_GB and small and big["seq1024"]["s_per_example"] <= 3 * small["seq1024"]["s_per_example"]:
            choice = CANDIDATES[1]
        else:
            choice = ok[0][0]
    res["choice"] = choice
    res["rule"] = f"largest candidate with seq1024 peak VRAM <= {VRAM_BUDGET_GB} GB and step time <= 3x the 1.5B"
    config.MODELS.mkdir(parents=True, exist_ok=True)
    (config.MODELS / "bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    config.BASE_CHOICE.write_text(json.dumps({"base": choice, "from": "bench.json"}), encoding="utf-8")
    log(f"choice: {choice}")
    return res
