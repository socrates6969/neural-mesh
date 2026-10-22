"""`python -m llm external`: build data/lifelong/sft_external.jsonl, TRAIN-ONLY extra data for the nightly QLoRA.

Open-source datasets fetched through the Hugging Face datasets-server REST API (random pages, no extra packages):
MetaMathQA (math worked solutions), Magicoder-OSS-Instruct-75K (code), camel-ai/physics (physics Q&A).
All text is scrubbed with ingest.scrub; rows with secret-class hits are dropped. Refreshed by the nightly script when
older than 7 days. These rows never enter the held-out split, so nightly evaluation stays comparable.
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


def _get(url: str) -> dict:
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            time.sleep(2 + 3 * attempt)
    return {}


def oss_rows(rng: random.Random) -> list[dict]:
    out = []
    for ds, n, fi, fr, label in SOURCES:
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


def build(seed: int | None = None) -> dict:
    sys.path.insert(0, str(config.REPO))
    from ingest import scrub as S
    rng = random.Random(seed if seed is not None else int(time.time()))
    rows, dropped = [], 0
    for r in oss_rows(rng):
        si, sr = S.scrub(r["instruction"]), S.scrub(r["response"])
        if si.secret or sr.secret:
            dropped += 1
            continue
        h = hashlib.sha256((r["instruction"] + "\x00" + r["response"]).encode("utf-8")).hexdigest()[:16]
        rows.append({"id": "ext:" + h, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "project": "external",
                     "session": "external", "source": r["source"], "agent": None, "model": None,
                     "instruction": si.text, "response": sr.text, "outcome": None, "judge_score": None,
                     "scrub_version": S.SCRUB_VERSION})
    tmp = config.EXTERNAL.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    tmp.replace(config.EXTERNAL)
    by = {}
    for r in rows:
        by[r["source"]] = by.get(r["source"], 0) + 1
    print("external:", len(rows), "rows", by, "| dropped (secrets):", dropped, flush=True)
    return {"rows": len(rows), "by_source": by, "dropped": dropped}


def stale(days: float = 7.0) -> bool:
    return not config.EXTERNAL.exists() or time.time() - config.EXTERNAL.stat().st_mtime > days * 86400


if __name__ == "__main__":
    build()