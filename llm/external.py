"""`python -m llm external [--eval] [--if-stale]`: open-source data for the nightly QLoRA, fetched through the
Hugging Face datasets-server REST API (random pages, no extra packages):
MetaMathQA (math worked solutions), Magicoder-OSS-Instruct-75K (code), camel-ai/physics (physics Q&A).

* data/lifelong/sft_external.jsonl       TRAIN-ONLY rows (refreshed weekly). Never in the held-out split.
* data/lifelong/sft_eval_external.jsonl  a FIXED set of ~1000 held-out questions with reference answers, used only by
                                         the nightly pairwise judge (built once, fixed seed, so nights are comparable).
The two sets never share a row (each build excludes the other's ids). All text is scrubbed with ingest.scrub; rows
with secret-class hits are dropped.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import urllib.parse
import urllib.request

from . import config

API = "https://datasets-server.huggingface.co/rows?dataset=%s&config=default&split=train&offset=%d&length=100"
SOURCES = [  # (dataset, rows to take, instruction field, response field, label)
    ("meta-math/MetaMathQA", 3000, "query", "response", "oss:metamathqa"),
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", 3000, "problem", "solution", "oss:magicoder"),
    ("camel-ai/physics", 1500, "message_1", "message_2", "oss:camel-physics"),
]
EVAL_SOURCES = [(ds, n, fi, fr, label) for (ds, _, fi, fr, label), n in zip(SOURCES, (400, 400, 200))]


def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(2 + 3 * attempt)
    return {}


def _ids(p) -> set:
    if not p.exists():
        return set()
    return {json.loads(l)["id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}


def oss_rows(rng: random.Random, sources) -> list[dict]:
    out = []
    for ds, n, fi, fr, label in sources:
        q = urllib.parse.quote(ds, safe="")
        total = int(_get(API % (q, 0)).get("num_rows_total") or 0)
        if not total:
            print("skip", ds, "(unreachable)", flush=True)
            continue
        offsets = rng.sample(range(0, max(1, total - 100), 100), k=min(n // 100, max(1, total // 100)))
        got = 0
        for off in offsets:
            for row in (_get(API % (q, off)).get("rows") or []):
                r = row["row"]
                ins, resp = str(r.get(fi) or "").strip(), str(r.get(fr) or "").strip()
                if 20 < len(ins) < 6000 and 20 < len(resp) < 8000:
                    out.append({"source": label, "instruction": ins, "response": resp})
                    got += 1
        print(ds, got, "rows", flush=True)
    return out


def build(seed: int | None = None, out=None, sources=None, exclude=None) -> dict:
    """Train set by default; `exclude` ids (default: the eval set) are skipped so train and eval never overlap."""
    sys.path.insert(0, str(config.REPO))
    from ingest import scrub as S
    out = out or config.EXTERNAL
    exclude = _ids(config.EVAL_EXTERNAL) if exclude is None else exclude
    rng = random.Random(seed if seed is not None else int(time.time()))
    rows, dropped, seen = [], 0, set()
    for r in oss_rows(rng, sources or SOURCES):
        si, sr = S.scrub(r["instruction"]), S.scrub(r["response"])
        if si.secret or sr.secret:
            dropped += 1
            continue
        rid = "ext:" + hashlib.sha256((r["instruction"] + "\x00" + r["response"]).encode("utf-8")).hexdigest()[:16]
        if rid in exclude or rid in seen:
            continue
        seen.add(rid)
        rows.append({"id": rid, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "project": "external",
                     "session": ("exteval:" + rid) if out == config.EVAL_EXTERNAL else "external", "source": r["source"],
                     "agent": None, "model": None, "instruction": si.text, "response": sr.text, "outcome": None,
                     "judge_score": None, "scrub_version": S.SCRUB_VERSION})
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    tmp.replace(out)
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print("%s: %d rows %s | dropped (secrets): %d" % (out.name, len(rows), by, dropped), flush=True)
    return {"rows": len(rows), "by_source": by, "dropped": dropped}


def build_eval(seed: int = 20261022) -> dict:
    """The fixed held-out question set for the pairwise judge (excludes every current training row)."""
    return build(seed=seed, out=config.EVAL_EXTERNAL, sources=EVAL_SOURCES, exclude=_ids(config.EXTERNAL))


def load_eval() -> list[dict]:
    if not config.EVAL_EXTERNAL.exists():
        return []
    return [json.loads(l) for l in config.EVAL_EXTERNAL.read_text(encoding="utf-8").splitlines() if l.strip()]


def stale(days: float = 7.0) -> bool:
    return not config.EXTERNAL.exists() or time.time() - config.EXTERNAL.stat().st_mtime > days * 86400


if __name__ == "__main__":
    build_eval() if "--eval" in sys.argv else build()