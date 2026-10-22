"""Benchmarks for agent training (batch 1: math, physics, coding, computer-science, neural-networks).

Every task: {id, field, source, prompt, kind, answer, ...}. kind in {int, numeric, mcq, code}.
Open data comes from the Hugging Face datasets-server JSON API (no extra packages); neural-network tasks are generated
with numpy so their answers are exact. Splits are fixed by a hash of the task id: "tune" (packs may learn from it)
and "held" (never shown to the pack writer; sliced per round so every round is scored on fresh tasks).
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"


def _rows(ds, cfg, split):
    out, off = [], 0
    while True:
        for attempt in range(4):
            try:
                with urllib.request.urlopen(API.format(ds=ds, cfg=cfg, split=split, off=off), timeout=60) as r:
                    page = json.load(r)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        rows = [x["row"] for x in page.get("rows", [])]
        out += rows
        off += len(rows)
        if not rows or off >= page.get("num_rows_total", 0):
            return out


def _is_int(s):
    return bool(re.fullmatch(r"-?\d+", (s or "").strip()))


def math_tasks():
    t = [{"id": "aime-" + str(r["id"]), "field": "math", "source": "AI-MO/aimo-validation-aime", "kind": "int",
          "prompt": r["problem"] + "\n\nThe answer is an integer.", "answer": str(int(r["answer"])), "ref": r.get("solution", "")}
         for r in _rows("AI-MO/aimo-validation-aime", "default", "train")]
    for r in _rows("HuggingFaceH4/MATH-500", "default", "test"):
        if str(r.get("level")) == "5" and _is_int(r["answer"]):
            t.append({"id": "math500-" + r["unique_id"], "field": "math", "source": "HuggingFaceH4/MATH-500", "kind": "int",
                      "prompt": r["problem"] + "\n\nThe answer is an integer.", "answer": str(int(r["answer"])), "ref": r.get("solution", "")})
    return t


HINTS = {"MCQ": "Exactly one option is correct.", "MCQ(multiple)": "One or more options may be correct; give all correct letters.",
         "Integer": "The answer is a non-negative integer.", "Numeric": "Give the numeric answer correct to 2 decimal places."}
KINDS = {"MCQ": "mcq", "MCQ(multiple)": "mcq", "Integer": "numeric", "Numeric": "numeric"}


def physics_tasks():
    t = []
    for r in _rows("daman1209arora/jeebench", "default", "test"):
        typ = r.get("type", "")
        if r.get("subject") != "phy" or typ not in KINDS:
            continue
        t.append({"id": "jee-phy-" + str(r["index"]), "field": "physics", "source": "jeebench " + str(r.get("description", "")),
                  "kind": KINDS[typ], "prompt": r["question"] + "\n\n" + HINTS[typ], "answer": str(r["gold"]).strip()})
    return t


def coding_tasks():
    return [{"id": r["task_id"].replace("/", "-"), "field": "coding", "source": "evalplus/humanevalplus", "kind": "code",
             "prompt": "Complete this Python function. Return the full function (with any imports) in one ```python block.\n\n```python\n"
                       + r["prompt"] + "```", "entry_point": r["entry_point"], "test": r["test"], "answer": ""}
            for r in _rows("evalplus/humanevalplus", "default", "test")]


def cs_tasks():
    t = []
    for r in _rows("evalplus/mbppplus", "default", "test"):
        try:
            examples = ast.literal_eval(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]
        except Exception:
            continue
        m = re.search(r"assert\s+(?:\w+\()*\s*(\w+)\(", examples[0]) if examples else None
        names = re.findall(r"def\s+(\w+)\s*\(", r.get("code", ""))
        name = names[-1] if names else (m.group(1) if m else None)
        if not name:
            continue
        t.append({"id": "mbpp-" + str(r["task_id"]), "field": "computer-science", "source": "evalplus/mbppplus", "kind": "code",
                  "prompt": r["prompt"] + "\nThe function must be named `" + name + "`. Example: " + examples[0] +
                            "\nReturn the full Python solution (with any imports) in one ```python block. Hidden tests include edge cases.",
                  "entry_point": name, "test": r["test"], "answer": ""})
    return t


def nn_tasks(n=160, seed=7):
    """Generated neural-network questions with exact answers (shapes, parameter counts, gradients, optimiser steps, memory)."""
    import numpy as np
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 8
        if k == 0:
            h, w, c, f = rng.choice([32, 64, 96, 128, 224]), rng.choice([32, 64, 96]), rng.choice([3, 16, 32]), rng.choice([8, 16, 64])
            ks, s, p = rng.choice([1, 3, 5, 7]), rng.choice([1, 2]), rng.choice([0, 1, 2, 3])
            oh, ow = (h + 2 * p - ks) // s + 1, (w + 2 * p - ks) // s + 1
            q = ("A Conv2d layer has in_channels=%d, out_channels=%d, kernel_size=%d, stride=%d, padding=%d, bias=True. "
                 "Input shape is (N, %d, %d, %d). How many output elements are there per sample (C*H*W)?") % (c, f, ks, s, p, c, h, w)
            a = f * oh * ow
        elif k == 1:
            dims = [rng.choice([16, 32, 64, 100, 128, 256, 512, 784]) for _ in range(rng.randint(3, 5))]
            a = sum(dims[j] * dims[j + 1] + dims[j + 1] for j in range(len(dims) - 1))
            q = "An MLP has layer sizes %s (fully connected, every layer has a bias). How many trainable parameters does it have?" % dims
        elif k == 2:
            d, heads, L = rng.choice([256, 512, 768, 1024]), rng.choice([4, 8, 16]), rng.choice([128, 512, 1024, 2048])
            a = 4 * d * d + 4 * d
            q = ("A multi-head self-attention block (d_model=%d, %d heads, sequence length %d) has Q, K, V and output projections, "
                 "each a d_model x d_model linear layer with bias. How many parameters does the block have?") % (d, heads, L)
        elif k == 3:
            v, d, L = rng.choice([30000, 32000, 50257, 65536]), rng.choice([512, 768, 1024]), rng.choice([512, 1024, 2048])
            a = v * d + L * d
            q = ("A transformer has a token embedding table of vocab %d and dimension %d, plus a learned positional embedding for %d "
                 "positions. How many parameters are in these two embeddings together?") % (v, d, L)
        elif k == 4:
            x = np.round(np.array([rng.uniform(-3, 3) for _ in range(4)]), 2)
            y, j = rng.randrange(4), rng.randrange(4)
            p_ = np.exp(x - x.max()); p_ /= p_.sum(); g = p_.copy(); g[y] -= 1
            q = ("Logits z = %s, true class index %d (0-based). With softmax followed by cross-entropy loss, what is dL/dz[%d]? "
                 "Give 4 decimals.") % (x.tolist(), y, j)
            a = round(float(g[j]), 4)
        elif k == 5:
            w1, b1, w2, b2, xv, t_ = [round(rng.uniform(-2, 2), 2) for _ in range(6)]
            hpre = w1 * xv + b1; hv = max(hpre, 0.0); yhat = w2 * hv + b2; dy = 2 * (yhat - t_)
            a = round(dy * w2 * (1.0 if hpre > 0 else 0.0) * xv, 4)
            q = ("Network: h = ReLU(w1*x + b1), y = w2*h + b2, loss = (y - t)^2. With w1=%s, b1=%s, w2=%s, b2=%s, x=%s, t=%s, "
                 "what is dLoss/dw1? Give 4 decimals.") % (w1, b1, w2, b2, xv, t_)
        elif k == 6:
            lr, wd, steps = rng.choice([0.1, 0.05, 0.01]), rng.choice([0.0, 0.01, 0.1]), rng.randint(3, 6)
            w0, gc = round(rng.uniform(-1, 1), 3), round(rng.uniform(-1, 1), 3)
            w = w0
            for _ in range(steps):
                w = w - lr * (gc + wd * w)
            a = round(w, 4)
            q = ("Plain SGD with learning rate %s and L2 weight decay coefficient %s (gradient term wd*w added to the loss gradient). "
                 "A scalar weight starts at %s; the loss gradient is a constant %s at every step. What is the weight after %d steps? "
                 "Give 4 decimals.") % (lr, wd, w0, gc, steps)
        else:
            b, s_, d, layers = rng.choice([1, 8, 32]), rng.choice([512, 1024, 2048]), rng.choice([768, 1024, 4096]), rng.choice([12, 24, 32])
            a = 2 * b * s_ * d * layers * 2
            q = ("For KV caching in a decoder-only transformer with %d layers, d_model=%d, batch size %d, and %d cached tokens, "
                 "how many BYTES does the K and V cache take in fp16 (2 bytes per value), ignoring any overhead?") % (layers, d, b, s_)
        out.append({"id": "nn-gen-%d" % i, "field": "neural-networks", "source": "generated (numpy)",
                    "kind": "int" if isinstance(a, int) else "numeric", "prompt": q, "answer": str(a)})
    return out


BUILDERS = {"math": math_tasks, "physics": physics_tasks, "coding": coding_tasks, "computer-science": cs_tasks, "neural-networks": nn_tasks}


def split_of(task_id, tune_frac=0.35):
    h = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "tune" if h < tune_frac else "held"


def build_all(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for field, fn in BUILDERS.items():
        tasks = fn()
        for t in tasks:
            t["split"] = split_of(t["id"])
        random.Random(field).shuffle(tasks)          # fixed order -> per-round held slices are fixed in advance
        (out_dir / (field + ".jsonl")).write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in tasks) + "\n", encoding="utf-8")
        stats[field] = {"total": len(tasks), "tune": sum(t["split"] == "tune" for t in tasks), "held": sum(t["split"] == "held" for t in tasks)}
    return stats


def load(bench_dir, field):
    return [json.loads(l) for l in (Path(bench_dir) / (field + ".jsonl")).read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    import sys
    print(json.dumps(build_all(sys.argv[1]), indent=1))
