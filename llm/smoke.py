"""`python -m llm smoke --base NAME [--steps 30] [--out FILE]`: a REAL short QLoRA run on one candidate base through the
nightly code path (model.load_4bit trainable, model.encode at config.MAX_LEN, train_loop bs 1 x grad_accum 8, gradient
checkpointing), then the eval generation pass (model.generate_batch at bs 8: 256 new tokens for pairwise answers, 768 for the
deterministic battery). Peak VRAM is taken three ways: torch allocated, torch reserved, and whole-GPU used (nvidia-smi sampled
every 0.5 s, so the desktop's share is included). The result decides whether the base fits the night (llm/README.md).

Examples: the longest real rows first (worst case for activations), then a random real mix (sft.jsonl + sft_external.jsonl).
Prompts for the generation pass: the fixed open-source eval questions (sft_eval_external.jsonl), else held-out rows."""
from __future__ import annotations

import gc
import json
import os
import random
import subprocess
import threading
import time
from pathlib import Path

from . import config, data

N_TASK_PROMPTS, N_DET_PROMPTS = 32, 16
VRAM_CAP_GB = 7.0        # torch allocator cap: past it torch raises OOM instead of the Windows driver spilling into system RAM
MIN_FREE_RAM_MB = 600    # watchdog: exit (code 3) rather than push the machine into swap


def free_ram_mb() -> int | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullAvailPhys / 2**20)
    except (AttributeError, OSError):
        return None


class GpuSampler:
    """whole-GPU used MiB via nvidia-smi, sampled in a thread (max over the run)."""

    def __init__(self, every: float = 0.5):
        self.every, self.peak, self.idle, self._stop = every, 0, None, threading.Event()

    @staticmethod
    def used_mib() -> int | None:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None

    def __enter__(self):
        self.idle = self.used_mib()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            u = self.used_mib()
            if u:
                self.peak = max(self.peak, u)
            r = free_ram_mb()
            if r is not None and r < MIN_FREE_RAM_MB:
                print(f"WATCHDOG: free RAM {r} MB < {MIN_FREE_RAM_MB} MB (GPU {u} MiB): exiting", flush=True)
                os._exit(3)
            self._stop.wait(self.every)

    def __exit__(self, *exc):
        self._stop.set()
        self.t.join(timeout=5)


def examples(tok, n: int, seed: int = config.SEED) -> list[tuple[list[int], list[int]]]:
    from . import model as M
    rows = data.read_jsonl(config.SFT) + data.read_jsonl(config.EXTERNAL)
    if not rows:
        rows = data.synthetic_rows()
    random.Random(seed).shuffle(rows)
    enc = [M.encode(tok, r["instruction"], r["response"]) for r in rows[: n * 4]]
    enc.sort(key=lambda e: -len(e[0]))
    longest, rest = enc[: n // 3], enc[n // 3:]
    random.Random(seed).shuffle(rest)
    return longest + rest[: n - len(longest)]


def prompts() -> tuple[list[str], list[str]]:
    from .evaluate import det_items
    ev = data.read_jsonl(config.EVAL_EXTERNAL) or data.read_jsonl(config.SFT)
    random.Random(config.SEED).shuffle(ev)
    return [r["instruction"] for r in ev[:N_TASK_PROMPTS]], [i["prompt"] for i in det_items(N_DET_PROMPTS // 4)]


def run(base_name: str, steps: int = 30, grad_accum: int = 8, gen_bs: int = 8, out: Path | None = None, log=print) -> dict:
    import torch
    from . import model as M
    base = config.base_path(base_name)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 2**30 / total))
    tok = M.tokenizer(base)
    enc = examples(tok, steps * grad_accum)
    res = {"base": base_name, "max_len": config.MAX_LEN, "steps": steps, "bs": 1, "grad_accum": grad_accum,
           "examples": len(enc), "mean_tokens": round(sum(len(e[0]) for e in enc) / len(enc)), "max_tokens": max(len(e[0]) for e in enc),
           "gpu": torch.cuda.get_device_name(0), "vram_cap_gb": VRAM_CAP_GB,
           "fp32_upcast": M.FP32_UPCAST}
    with GpuSampler() as g:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        m = M.load_4bit(base, trainable=True)
        res["load_s"] = round(time.time() - t, 1)
        res["load_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        try:
            st = M.train_loop(m, enc, epochs=1, bs=1, grad_accum=grad_accum, max_steps=steps, log=log)
        except torch.OutOfMemoryError as e:     # does not fit under the cap: record it, that IS the result
            res.update(oom=True, oom_error=str(e)[:300], peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                       train_gpu_peak_mib=g.peak)
            log(f"{base_name}: OOM under the {VRAM_CAP_GB} GB cap")
            out = out or (config.MODELS / f"smoke_{base_name}.json")
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")
            return res
        res["train"] = st
        res["train_s_per_example"] = round(st["seconds"] / max(1, st["micro_batches"]), 3)
        res["train_gpu_peak_mib"] = g.peak
        res["gpu_idle_mib_before"] = g.idle
    log(f"{base_name} train: {st}; whole-GPU peak {g.peak} MiB (idle before {g.idle} MiB)")
    task, det = prompts()
    with GpuSampler() as g:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        ans = M.generate_batch(m, tok, task, bs=gen_bs)
        dt_task = time.time() - t
        t = time.time()
        M.generate_batch(m, tok, det, max_new=768, bs=gen_bs)
        dt_det = time.time() - t
        res["gen"] = {"bs": gen_bs, "task_items": len(task), "task_s_per_item": round(dt_task / len(task), 3),
                      "task_mean_answer_tokens": round(sum(len(tok(a)["input_ids"]) for a in ans) / len(ans)),
                      "det_items": len(det), "det_s_per_item": round(dt_det / len(det), 3),
                      "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                      "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3), "gpu_peak_mib": g.peak}
    log(f"{base_name} generation: {res['gen']}")
    res["estimate"] = estimate(res)
    log(f"{base_name} estimate: {res['estimate']}")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    out = out or (config.MODELS / f"smoke_{base_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def estimate(res: dict, n_pairs: int = 900, n_det: int = 80, n_loss: int = 200) -> dict:
    """the nightly at this base: examples trained in config.TRAIN_MINUTES, and the eval's GPU minutes (both models answer
    every pairwise item and battery item; the loss test is about one forward pass per example per model)."""
    g = res["gen"]
    gen_min = 2 * (n_pairs * g["task_s_per_item"] + n_det * g["det_s_per_item"]) / 60
    loss_min = 2 * n_loss * res["train_s_per_example"] / 3 / 60   # forward only ~ 1/3 of a training step
    return {"train_examples_in_budget": int(config.TRAIN_MINUTES * 60 / res["train_s_per_example"]),
            "eval_generation_min": round(gen_min, 1), "eval_loss_min": round(loss_min, 1), "n_pairs": n_pairs, "n_det": n_det}
