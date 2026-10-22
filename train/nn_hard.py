"""Harder neural-network tasks (numeric answers computed exactly from stated conventions).

The v1 NN set (conv output sizes, MLP param counts, ...) saturated at 20/20. These need correct modelling knowledge:
mixed-precision Adam memory with ZeRO, GQA/MoE parameter counts, transformer FLOPs under a stated convention,
KV-cache size, warmup+cosine LR, LayerNorm backward, dilated receptive fields, Chinchilla-optimal sizing,
label-smoothed cross-entropy. Every prompt states its convention so the answer is unique.
"""
from __future__ import annotations

import math
import random

import numpy as np


def _t(i, kind, prompt, ans, field="neural-networks"):
    return {"id": "nnh-%s-%d" % (kind, i), "field": field, "source": "generated-hard:" + kind, "kind": "numeric",
            "prompt": prompt + "\n\nGive the numeric answer (relative error under 1% is accepted).", "answer": repr(float(ans))}


def zero_memory(i, r):
    p = r.choice([1.3, 3, 7, 13, 34, 70]); n = r.choice([1, 4, 8, 16, 64]); stage = r.choice([0, 1, 2, 3])
    w, g, opt = 2.0 * p, 2.0 * p, 12.0 * p  # fp16 weights, fp16 grads, fp32 master + Adam m + v (4+4+4 bytes/param)
    per = (w if stage < 3 else w / n) + (g if stage < 2 else g / n) + (opt if stage < 1 else opt / n)
    return _t(i, "zero", ("Mixed-precision Adam training of a %s-billion-parameter model on %d GPUs with ZeRO stage %d "
            "(stage 0 = plain data parallel; stage 1 shards optimizer states; stage 2 also gradients; stage 3 also weights). "
            "Per parameter: fp16 weights (2 B), fp16 gradients (2 B), fp32 master weights + Adam m + v (12 B). Ignore "
            "activations, buffers and fragmentation. How many GB (1 GB = 1e9 bytes) of model-state memory per GPU?") % (p, n, stage), per)


def gqa_params(i, r):
    d = r.choice([2048, 3072, 4096, 5120, 8192]); hd = 128; nh = d // hd; nkv = r.choice([1, 2, 4, 8])
    ans = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    return _t(i, "gqa", ("A grouped-query attention layer has d_model=%d, %d query heads, %d key/value heads, head_dim=%d, "
            "no biases. Count the parameters of the Q, K, V and output projections only.") % (d, nh, nkv, hd), ans)


def moe_params(i, r):
    d = r.choice([2048, 4096, 6144]); ff = r.choice([4, 8]) * d // 3 * 2; e = r.choice([8, 16, 64]); k = r.choice([1, 2])
    per = 3 * d * ff  # SwiGLU: gate, up, down
    total, active = e * per + d * e, k * per + d * e
    which = r.choice(["total", "active"])
    return _t(i, "moe", ("One MoE feed-forward layer: d_model=%d, %d experts, top-%d routing, each expert is a SwiGLU MLP with "
            "hidden size %d (three bias-free matrices: gate and up d_model x hidden, down hidden x d_model), plus a bias-free "
            "router d_model x num_experts. How many %s parameters (router included; 'active' = used for one token)?")
            % (d, e, k, ff, which), total if which == "total" else active)


def fwd_flops(i, r):
    L = r.choice([12, 24, 32, 48]); d = r.choice([768, 1024, 2048, 4096]); s = r.choice([512, 1024, 2048, 4096])
    per_layer = 2 * (4 * d * d) + 2 * (2 * d * 4 * d) + 2 * (2 * s * d)  # QKVO, MLP 4x, scores+values (per token)
    return _t(i, "flops", ("Forward pass FLOPs per token of a decoder-only transformer: %d layers, d_model=%d, MLP hidden 4*d_model "
            "(two matrices), full attention over context %d. Count a multiply-add as 2 FLOPs; count Q,K,V,O projections, the "
            "two MLP matrices, and the attention score (QK^T) and value (AV) matmuls averaged as if every token attends to all "
            "%d positions; ignore softmax, norms, embeddings and the LM head. Answer in GFLOPs (1e9).") % (L, d, s, s), L * per_layer / 1e9)


def kv_cache(i, r):
    L = r.choice([32, 40, 80]); nkv = r.choice([8, 4, 2]); hd = 128; b = r.choice([1, 8, 32]); s = r.choice([4096, 8192, 32768])
    by = r.choice([(2, "bf16"), (1, "fp8")])
    return _t(i, "kv", ("KV cache size: %d layers, %d KV heads, head_dim %d, batch %d, sequence length %d, %s storage (%d byte(s) per "
            "value). Keys and values both cached. Answer in GiB (2^30 bytes).") % (L, nkv, hd, b, s, by[1], by[0]),
              2 * L * nkv * hd * b * s * by[0] / 2**30)


def lr_sched(i, r):
    peak = r.choice([3e-4, 6e-4, 1e-3]); mn = peak / 10; W = r.choice([500, 1000, 2000]); T = r.choice([10000, 50000, 100000])
    s = r.randrange(0, T + 1)
    lr = peak * s / W if s < W else mn + 0.5 * (peak - mn) * (1 + math.cos(math.pi * (s - W) / (T - W)))
    return _t(i, "lr", ("Learning rate schedule: linear warmup from 0 to %g over the first %d steps (lr = peak*step/%d for step < %d), "
            "then cosine decay from %g to %g reaching the minimum at step %d: lr = min + 0.5*(peak-min)*(1+cos(pi*(step-%d)/(%d-%d))). "
            "What is the learning rate at step %d?") % (peak, W, W, W, peak, mn, T, W, T, W, s), lr)


def ln_backward(i, r):
    rs = np.random.RandomState(r.randrange(10**6))
    x = np.round(rs.randn(5), 2); gam = np.round(rs.rand(5) + 0.5, 2); dy = np.round(rs.randn(5), 2); j = r.randrange(5)
    mu, var = x.mean(), x.var(); xh = (x - mu) / np.sqrt(var)
    dxh = dy * gam
    dx = (dxh - dxh.mean() - xh * (dxh * xh).mean()) / np.sqrt(var)
    return _t(i, "lnbwd", ("LayerNorm over a single vector x=%s with eps=0, gamma=%s, beta=0 (biased variance). Upstream gradient "
            "dL/dy=%s. Give dL/dx[%d] (0-indexed).") % (list(x), list(gam), list(dy), j), dx[j])


def receptive(i, r):
    n = r.randrange(3, 7); layers = [(r.choice([3, 5, 7]), r.choice([1, 1, 2]), r.choice([1, 1, 2, 4])) for _ in range(n)]
    rf, jump = 1, 1
    for k, s, dil in layers:
        rf += (k - 1) * dil * jump; jump *= s
    desc = "; ".join("kernel %d, stride %d, dilation %d" % l for l in layers)
    return _t(i, "rf", "Receptive field (in input pixels, 1-D) of one output unit after this stack of conv layers, in order: %s." % desc, rf)


def chinchilla(i, r):
    c = r.choice([1e21, 3e21, 1e22, 6e22, 1e23, 5e23, 1e24])
    n = math.sqrt(c / 120)  # C = 6 N D, D = 20 N
    which = r.choice(["parameters", "tokens"])
    return _t(i, "chin", ("Using C = 6*N*D training FLOPs and the Chinchilla rule D = 20*N tokens per parameter, what is the "
            "compute-optimal number of %s for a budget of C = %.0e FLOPs? Answer in billions.") % (which, c),
              (n if which == "parameters" else 20 * n) / 1e9)


def ls_ce(i, r):
    rs = np.random.RandomState(r.randrange(10**6))
    k = r.choice([4, 5, 6]); z = np.round(rs.randn(k) * 2, 2); y = r.randrange(k); eps = r.choice([0.1, 0.2])
    p = np.exp(z - z.max()); p /= p.sum(); q = np.full(k, eps / k); q[y] += 1 - eps
    return _t(i, "lsce", ("Cross-entropy with label smoothing eps=%g over %d classes (target distribution: 1-eps on the true class "
            "plus eps/%d on every class). Logits %s, true class index %d. Natural-log loss value?") % (eps, k, k, list(z), y),
              float(-(q * np.log(p)).sum()))


GENS = [zero_memory, gqa_params, moe_params, fwd_flops, kv_cache, lr_sched, ln_backward, receptive, chinchilla, ls_ce]


def tasks(n_per=18, seed=7):
    r = random.Random(seed)
    return [g(i, r) for g in GENS for i in range(n_per)]