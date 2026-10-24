"""QLoRA model plumbing: 4-bit base + LoRA adapter, example encoding with prompt masking, a plain training loop,
per-example held-out loss, and greedy generation. Heavy imports (torch/transformers/peft) live here only."""
from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import config

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
SYSTEM = "You are a concise, careful coding assistant."
FP32_UPCAST = os.environ.get("MESH_LLM_FP32_UPCAST", "0") == "1"   # 1 = the old prepare_model_for_kbit_training path


def tokenizer(base: Path):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_4bit(base: Path, adapter: Path | None = None, trainable: bool = False):
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                           bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0})
    m.config.use_cache = False
    if trainable:
        if FP32_UPCAST:
            m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
        else:   # same as prepare_model_for_kbit_training minus its fp32 upcast of every non-quantized weight: for a 151k
            # vocabulary that upcast alone costs ~0.8 GB (3B) to ~1.5 GB (4B) of VRAM, and no LoRA target is affected by it
            for prm in m.parameters():
                prm.requires_grad_(False)
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
        if adapter is not None:
            m = PeftModel.from_pretrained(m, adapter, is_trainable=True)
        else:
            m = get_peft_model(m, LoraConfig(**LORA))
    elif adapter is not None:
        m = PeftModel.from_pretrained(m, adapter)
    return m


def prompt_text(tok, instruction: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": instruction}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(tok, instruction: str, response: str, max_len: int = config.MAX_LEN) -> tuple[list[int], list[int]]:
    """input_ids, labels (prompt tokens = -100). Over-long examples keep the END of the prompt and as much of the
    response as fits (the response is what we learn; at least half the budget goes to it)."""
    p = tok(prompt_text(tok, instruction), add_special_tokens=False)["input_ids"]
    r = tok(response + tok.eos_token, add_special_tokens=False)["input_ids"]
    r = r[: max(max_len // 2, max_len - len(p))]
    p = p[-(max_len - len(r)):] if len(p) + len(r) > max_len else p
    return p + r, [-100] * len(p) + r


def _batches(encoded, bs):
    for i in range(0, len(encoded), bs):
        chunk = encoded[i:i + bs]
        L = max(len(x[0]) for x in chunk)
        ids = torch.full((len(chunk), L), 0, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, (a, b) in enumerate(chunk):
            ids[j, :len(a)] = torch.tensor(a)
            lab[j, :len(b)] = torch.tensor(b)
            att[j, :len(a)] = 1
        yield ids.cuda(), lab.cuda(), att.cuda()


def lm_loss(model, ids, lab, att):
    """mean token NLL over the labelled (response) tokens. At bs 1 the labelled tokens are the tail of the sequence, so only
    those positions get logits (logits_to_keep): the full-vocabulary fp32 logits of the prompt are never materialised."""
    n = int((lab[0] != -100).sum().item()) if ids.size(0) == 1 else 0
    if not n or n >= ids.size(1):
        return model(input_ids=ids, attention_mask=att, labels=lab).loss
    logits = model(input_ids=ids, attention_mask=att, logits_to_keep=n + 1).logits[:, :-1]
    tgt = lab[:, -n:]
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)


def train_loop(model, encoded: list[tuple[list[int], list[int]]], epochs: int = 2, lr: float = 2e-4, bs: int = 1,
               grad_accum: int = 8, seed: int = config.SEED, log=print, time_budget_s: float | None = None,
               max_steps: int | None = None) -> dict:
    """time_budget_s: stop at this wall-clock budget. After 20 optimizer steps the LR-schedule horizon is re-estimated
    from measured throughput so warmup+decay fit the budget (nightly: config.TRAIN_MINUTES). max_steps: hard stop after
    that many optimizer steps (smoke tests)."""
    import bitsandbytes as bnb
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    steps_total = max(1, math.ceil(len(encoded) * epochs / (bs * grad_accum)))
    horizon = [steps_total]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, horizon[0] // 10)) * max(0.05, 1 - s / horizon[0]))
    stopped = False
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, step, micro, losses = time.time(), 0, 0, []
    for ep in range(epochs):
        order = encoded[:]
        rng.shuffle(order)
        order.sort(key=lambda x: len(x[0]))  # length-bucketed micro-batches (bs>1) ...
        groups = [order[i:i + bs] for i in range(0, len(order), bs)]
        rng.shuffle(groups)                   # ... in random order
        for g in groups:
            ids, lab, att = next(_batches(g, bs))
            loss = lm_loss(model, ids, lab, att)
            (loss / grad_accum).backward()
            losses.append(loss.item())
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if time_budget_s and step == 20:
                    fit = int((time_budget_s / max(1e-6, time.time() - t0)) * step)
                    horizon[0] = max(step + 1, min(steps_total, fit))
                    log(f"  time budget {time_budget_s / 60:.0f} min -> schedule horizon {horizon[0]} of {steps_total} steps")
                if step % 10 == 0:
                    log(f"  step {step}/{horizon[0]} loss {sum(losses[-grad_accum:]) / grad_accum:.4f}")
                if time_budget_s and (time.time() - t0 > time_budget_s or step >= horizon[0] and horizon[0] < steps_total):
                    stopped = True
                    break
                if max_steps and step >= max_steps:
                    stopped = True
                    break
        log(f"epoch {ep + 1}/{epochs} mean loss {sum(losses[-len(groups):]) / max(1, len(groups)):.4f}")
        if stopped:
            log(f"time budget reached after {step} optimizer steps ({(time.time() - t0) / 60:.1f} min)")
            break
    if micro % grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
    return {"optimizer_steps": step, "micro_batches": micro, "seconds": round(time.time() - t0, 1), "time_budget_s": time_budget_s,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3), "final_loss": round(sum(losses[-16:]) / max(1, len(losses[-16:])), 4)}


@torch.no_grad()
def example_losses(model, encoded) -> list[tuple[float, int]]:
    """(sum of token NLL over the response, number of response tokens) per example."""
    model.eval()
    out = []
    for ids, lab, att in _batches(encoded, 1):
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        mask = tgt != -100
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.clamp(min=0).reshape(-1), reduction="none")
        nll = (nll.view_as(tgt) * mask).sum().item()
        out.append((nll, int(mask.sum().item())))
    return out


GEN_TOKEN_BUDGET = int(os.environ.get("MESH_LLM_GEN_TOKENS", "12288"))   # bs x (longest prompt + max_new) per generate() call


def length_batches(lengths: list[int], max_new: int, bs: int, budget: int = GEN_TOKEN_BUDGET) -> list[list[int]]:
    """indices grouped shortest-first; a batch grows up to bs while bs x (longest prompt + max_new) <= budget (>= 1 item)."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) >= bs or (len(cur) + 1) * (lengths[i] + max_new) > budget):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.no_grad()
def generate_batch(model, tok, instructions: list[str], max_new: int = 256, bs: int = 8, max_prompt: int = 1536) -> list[str]:
    """greedy answers in input order. Prompts are length-sorted and batched under a token budget (left padding), so short
    prompts share large batches and long ones never blow the KV cache; greedy decoding makes the order irrelevant."""
    model.eval()
    tok.padding_side = "left"
    enc = [tok(prompt_text(tok, x), add_special_tokens=False)["input_ids"][-max_prompt:] for x in instructions]
    out: list[str] = [""] * len(enc)
    for idx in length_batches([len(e) for e in enc], max_new, bs):
        ids = [enc[i] for i in idx]
        L = max(len(x) for x in ids)
        inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids], device="cuda")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
        g = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        for i, s in zip(idx, g):
            out[i] = tok.decode(s[L:], skip_special_tokens=True).strip()
    tok.padding_side = "right"
    return out


@torch.no_grad()
def generate(model, tok, instruction: str, max_new: int = 256) -> tuple[str, float, int]:
    """greedy answer, mean token log-prob of the answer (a confidence signal), number of answer tokens."""
    model.eval()
    enc = tok(prompt_text(tok, instruction), return_tensors="pt", add_special_tokens=False).to("cuda")
    enc = {k: v[:, -3072:] for k, v in enc.items()}
    g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, output_scores=True, return_dict_in_generate=True,
                       pad_token_id=tok.pad_token_id, use_cache=True)
    seq = g.sequences[0, enc["input_ids"].shape[1]:]
    lps = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, seq)]
    return tok.decode(seq, skip_special_tokens=True).strip(), (sum(lps) / len(lps) if lps else -99.0), len(seq)


@torch.no_grad()
def sample_batch(model, tok, instructions: list[str], k: int = 4, max_new: int = 768, temperature: float = 0.8, top_p: float = 0.95,
                 bs: int = 8, max_prompt: int = 1536) -> list[str]:
    """k sampled answers per instruction (rejection sampling, llm/selftrain.py), flat list in instruction order."""
    model.eval()
    tok.padding_side = "left"
    flat = [x for x in instructions for _ in range(k)]
    out = []
    for i in range(0, len(flat), bs):
        ids = [tok(prompt_text(tok, x), add_special_tokens=False)["input_ids"][-max_prompt:] for x in flat[i:i + bs]]
        L = max(len(x) for x in ids)
        inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids], device="cuda")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
        g = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                           top_p=top_p, pad_token_id=tok.pad_token_id, use_cache=True)
        out += [tok.decode(s[L:], skip_special_tokens=True).strip() for s in g]
    tok.padding_side = "right"
    return out
