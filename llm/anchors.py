"""Self-distillation anchors against catastrophic forgetting (2026-10-12).

Training only on session transcripts made candidate 20261011-014210 lose general reasoning (battery math 8->3, physics 8->1,
research 9->6 vs the base). Anchors are general questions whose TARGET is the base model's own answer (served by Ollama as
`mesh-llm:base`), so each round also rehearses what the base already does well, including the strict `ANSWER:` format.

* Sources (both MIT-licensed, downloaded with `datasets`): GSM8K `main` train (math word problems, gold numeric answer) and
  MMLU `validation` for science/maths subjects (4-way multiple choice, gold letter). Never the battery, never MMLU test.
* Rejection-sampled: an anchor is kept only if the base answer ends with a parsable `ANSWER:` line that matches the gold
  answer, so a target is never a wrong answer.
* Leakage guard: any candidate whose 8-gram word shingles overlap a bench battery prompt (all splits) is dropped.
* Output: data/lifelong/anchors.jsonl {id, instruction, response, source, license, gold}; deterministic order and ids.
`python -m llm anchors [--n N]` builds it; train.py mixes ANCHOR_RATIO x len(new) of them into every round.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request

from . import config

FORMAT = "\n\nThink step by step. Give the final answer on the last line in the form `ANSWER: <answer>`."
MMLU_SUBJECTS = ("high_school_physics", "college_physics", "conceptual_physics", "astronomy", "electrical_engineering",
                 "high_school_chemistry", "college_chemistry", "high_school_mathematics", "college_mathematics",
                 "elementary_mathematics", "high_school_statistics", "computer_security", "college_computer_science")
OLLAMA = "http://localhost:11434/api/generate"
BASE_TAG = "mesh-llm:base"
LICENSES = {"gsm8k": "MIT (openai/gsm8k)", "mmlu": "MIT (cais/mmlu)"}
_ANS = re.compile(r"ANSWER:\s*\**\s*([^\n*`]+)", re.I)


def path():
    return config.REPO / "data" / "lifelong" / "anchors.jsonl"


def _shingles(text: str, n: int = 8) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def battery_shingles() -> set[tuple[str, ...]]:
    out: set = set()
    for f in sorted((config.REPO / "bench").glob("battery*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for it in (d["items"] if isinstance(d, dict) else d):
            if isinstance(it, dict) and "prompt" in it:
                out |= _shingles(it["prompt"])
    return out


def leaks(prompt: str, bank: set) -> bool:
    return bool(_shingles(prompt) & bank)


def final_answer(text: str) -> str | None:
    m = _ANS.findall(text or "")
    return m[-1].strip().rstrip(".") if m else None


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    try:
        return float(m.group(0).replace(",", "")) if m else None
    except ValueError:
        return None


def correct(source: str, got: str | None, gold: str) -> bool:
    if got is None:
        return False
    if source == "gsm8k":
        a, b = _num(got), _num(gold)
        return a is not None and b is not None and abs(a - b) < 1e-6
    return got.strip().upper()[:1] == gold.upper()


def candidates(seed: int = config.SEED) -> list[dict]:
    from datasets import load_dataset
    out = []
    for r in load_dataset("openai/gsm8k", "main", split="train"):
        gold = r["answer"].split("####")[-1].strip()
        out.append({"source": "gsm8k", "q": r["question"].strip(), "gold": gold})
    for subj in MMLU_SUBJECTS:
        for r in load_dataset("cais/mmlu", subj, split="validation"):
            opts = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r["choices"]))
            out.append({"source": "mmlu", "q": f"{r['question'].strip()}\n\n{opts}\n\nAnswer with the letter of the correct option.",
                        "gold": "ABCD"[r["answer"]], "subject": subj})
    random.Random(seed).shuffle(out)
    return out


def generate(prompt: str, timeout: int = 180) -> str:
    body = json.dumps({"model": BASE_TAG, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 768, "seed": 0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["response"]


def build(n: int = 320, per_source: float = 0.5, log=print, gen=generate, cands=None) -> dict:
    """Keep up to n anchors (about half GSM8K, half MMLU), rejection-sampled and leak-checked. Idempotent and resumable:
    each kept row is appended immediately; rows already in anchors.jsonl are kept and not regenerated; the file is
    rewritten sorted and de-duplicated at the end."""
    p = path()
    have = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); have[r["id"]] = r
    rej_p = p.with_name("anchors_rejected.txt")          # ids whose base answer was wrong: never regenerated on resume
    rejected = set(rej_p.read_text(encoding="utf-8").split()) if rej_p.exists() else set()
    bank = battery_shingles()
    quota = {"gsm8k": int(round(n * per_source)), "mmlu": n - int(round(n * per_source))}
    kept = {s: sum(1 for r in have.values() if r["source"] == s) for s in quota}
    stats = {"tried": 0, "leak": 0, "wrong": 0, "kept_new": 0}
    for c in (cands if cands is not None else candidates()):
        s = c["source"]
        if kept[s] >= quota[s]:
            if all(kept[k] >= quota[k] for k in quota):
                break
            continue
        instr = c["q"] + FORMAT
        rid = "anc-" + hashlib.sha256(instr.encode()).hexdigest()[:16]
        if rid in have or rid in rejected:
            continue
        if leaks(c["q"], bank):
            stats["leak"] += 1
            continue
        stats["tried"] += 1
        resp = gen(instr)
        if not correct(s, final_answer(resp), c["gold"]):
            stats["wrong"] += 1
            rejected.add(rid)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(rej_p, "a", encoding="utf-8") as f:
                f.write(rid + "\n")
            continue
        have[rid] = {"id": rid, "instruction": instr, "response": resp.strip(), "source": s, "license": LICENSES[s], "gold": c["gold"]}
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:     # append as we go: an interrupted build resumes instead of starting over
            f.write(json.dumps(have[rid], ensure_ascii=False) + "\n")
        kept[s] += 1
        stats["kept_new"] += 1
        if stats["tried"] % 25 == 0:
            log(f"anchors: tried {stats['tried']}, kept {sum(kept.values())}/{n}, wrong {stats['wrong']}, leak {stats['leak']}")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(have.values(), key=lambda r: r["id"])
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    stats.update(total=len(rows), by_source={k: sum(1 for r in rows if r["source"] == k) for k in quota})
    log(f"anchors: {stats}")
    return stats


def load() -> list[dict]:
    p = path()
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample(k: int, seed: int) -> list[dict]:
    rows = load()
    if k <= 0 or not rows:
        return []
    return random.Random(seed).sample(rows, min(k, len(rows)))
