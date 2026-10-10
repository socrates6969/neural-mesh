"""QLoRA model plumbing: 4-bit base + LoRA adapter, example encoding with prompt masking, a plain training loop,
per-example held-out loss, and greedy generation. Heavy imports (torch/transformers/peft) live here only."""
from __future__ import annotations

import math
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
        m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=True)
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


def train_loop(model, encoded: list[tuple[list[int], list[int]]], epochs: int = 2, lr: float = 2e-4, bs: int = 1,
               grad_accum: int = 8, seed: int = config.SEED, log=print) -> dict:
    import bitsandbytes as bnb
    torch.manual_seed(seed)
    rng = random.Random(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    steps_total = max(1, math.ceil(len(encoded) * epochs / (bs * grad_accum)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps_total // 10)) * max(0.05, 1 - s / steps_total))
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
            out = model(input_ids=ids, attention_mask=att, labels=lab)
            (out.loss / grad_accum).backward()
            losses.append(out.loss.item())
            micro += 1
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    log(f"  step {step}/{steps_total} loss {sum(losses[-grad_accum:]) / grad_accum:.4f}")
        log(f"epoch {ep + 1}/{epochs} mean loss {sum(losses[-len(groups):]) / max(1, len(groups)):.4f}")
    if micro % grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
    return {"optimizer_steps": step, "micro_batches": micro, "seconds": round(time.time() - t0, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3), "final_loss": round(sum(losses[-16:]) / max(1, len(losses[-16:])), 4)}


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


@torch.no_grad()
def generate_batch(model, tok, instructions: list[str], max_new: int = 256, bs: int = 8, max_prompt: int = 1536) -> list[str]:
    """greedy answers, left-padded batches (much faster than one at a time on 4-bit weights)."""
    model.eval()
    tok.padding_side = "left"
    out = []
    for i in range(0, len(instructions), bs):
        prompts = [prompt_text(tok, x) for x in instructions[i:i + bs]]
        ids = [tok(p, add_special_tokens=False)["input_ids"][-max_prompt:] for p in prompts]
        L = max(len(x) for x in ids)
        inp = torch.tensor([[tok.pad_token_id] * (L - len(x)) + x for x in ids], device="cuda")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
        g = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, use_cache=True)
        out += [tok.decode(s[L:], skip_special_tokens=True).strip() for s in g]
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
