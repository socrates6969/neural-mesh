"""Memory with usefulness credit.

* Ledger ``memory/items.jsonl`` (rewritten atomically): one row per knowledge chunk (decision / fix / root cause /
  convention / lesson), keyed by a hash of its normalised text, so a chunk re-derived in a later session REINFORCES the
  same item instead of duplicating it.
* Retrieval: ``python -m ingest recall "<query>" --session <id>`` returns the top-k items (cosine over the mesh's default
  HashEmbedding, weighted by value) and appends a retrieval event to ``memory/retrievals.jsonl``.
* Credit: on each run, every retrieval whose session now has outcomes is credited with the mean reward of that
  session's outcomes after the retrieval (pos += r, neg += 1 - r). Counts decay with a half-life, so old evidence fades.
* value = (1 + pos) / (2 + pos + neg)  (Beta-mean, prior 0.5). Seen-again reinforcement adds +0.5 pos.
* Tiered forgetting (RD-LOG lessons: forget by measured low value, never by age alone when the item is used):
  prune when retrieved >= 3 times with value < 0.3; prune never-retrieved single-sighting items older than 180 days;
  cap the active set at ``mem_max_active`` by value. A pruned item that is seen again is revived.
* Export: active items go to a neural-mesh MeshStore (``memory/meshstore``, namespace ``lifelong``) through
  ``meshstore put-corpus`` with HashEmbedding vectors (Python port of ``knowledge::hash_embed``). Ids there are
  ``lifelong:<row>``; appends are incremental, a prune rebuilds the namespace.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .extract import parse_ts
from .util import append_jsonl, atomic_write_json, atomic_write_text, iter_jsonl, read_json, sha16

HALF_LIFE_DAYS = 90.0
PRUNE_MIN_USES = 3
PRUNE_VALUE = 0.3
COLD_DAYS = 180.0


def fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hash_embed(text: str, dim: int) -> list[float]:
    """Port of neural_mesh::knowledge::hash_embed for a task whose features are ``text.split_whitespace()`` (HashEmbedding::embed)."""
    v = [0.0] * dim
    for f in text.split():
        h = fnv1a(f)
        v[h % dim] += 1.0 if ((h >> 40) & 1) == 0 else -1.0
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def embed_text(text: str, dim: int) -> list[float]:
    """Canonical text for embedding: lowercase alphanumeric tokens (so punctuation does not split the hash space)."""
    return hash_embed(" ".join(re.findall(r"[a-z0-9]+", text.lower())), dim)


def norm_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _now() -> float:
    return datetime.now(timezone.utc).timestamp()


class MemoryStore:
    NEAR_DUP = 0.92  # MiniLM cosine at/above which a new chunk reinforces an existing item instead of adding one

    def __init__(self, cfg, vs=None):
        self.cfg = cfg
        self.vs = vs if (vs is not None and vs.available()) else None
        self.dir: Path = cfg.memory_dir
        self.items_path = self.dir / "items.jsonl"
        self.retr_path = self.dir / "retrievals.jsonl"
        self.meta_path = self.dir / "meta.json"
        self.items: dict[str, dict] = {r["id"]: r for r in iter_jsonl(self.items_path) if r.get("id")}
        self.meta = read_json(self.meta_path, {"seq": 0, "credited": [], "exported": []})
        self.stats = {"new": 0, "reinforced": 0, "near_dup_merged": 0, "revived": 0, "credited_retrievals": 0, "pruned": 0, "exported": 0,
                      "rebuilt": False, "space": "minilm-384" if self.vs else "hash-%d" % cfg.mem_dim}
        self._nd_ids: list[str] | None = None
        self._nd_mat = None

    def _near_dup(self, text: str) -> str | None:
        if not self.vs:
            return None
        import numpy as np
        if self._nd_ids is None:
            act = [it for it in self.items.values() if it["status"] == "active"]
            self.vs.ensure([it["text"] for it in act])
            self._nd_ids = [it["id"] for it in act]
            m = self.vs.matrix([it["text"] for it in act])
            self._nd_mat = m if m is not None else np.zeros((0, 384), "<f4")
        v = self.vs.get(text)
        if v is None or not self._nd_ids:
            return None
        sims = self._nd_mat @ v
        j = int(np.argmax(sims))
        return self._nd_ids[j] if sims[j] >= self.NEAR_DUP else None

    def _nd_add(self, iid: str, text: str) -> None:
        if self.vs and self._nd_ids is not None:
            import numpy as np
            v = self.vs.get(text)
            if v is not None:
                self._nd_ids.append(iid)
                self._nd_mat = np.vstack([self._nd_mat, v[None, :]])

    # ---------------------------------------------------------------- ingestion
    def add(self, text: str, tag: str, project: str, session: str, source_id: str, ts: str, source_score: float) -> None:
        iid = "mem:" + sha16(norm_key(text))
        it = self.items.get(iid)
        if it is None:
            dup = self._near_dup(text)
            if dup is not None:
                it, iid = self.items[dup], dup
                self.stats["near_dup_merged"] += 1
        if it is None:
            self.meta["seq"] += 1
            self.items[iid] = {"id": iid, "seq": self.meta["seq"], "text": text, "tag": tag, "project": project, "sessions": [session],
                               "sources": [source_id], "ts_first": ts, "ts_last": ts, "seen": 1, "retrievals": 0,
                               "pos": round(0.5 * source_score, 4), "neg": round(0.5 * (1 - source_score), 4), "decay_ts": ts,
                               "value": 0.0, "status": "active", "vec_key": None}
            if self.vs:
                from .vectors import vec_key
                self.items[iid]["vec_key"] = vec_key(text)
            self._nd_add(iid, text)
            self.stats["new"] += 1
            return
        if source_id in it["sources"]:
            return  # idempotent: the same source never reinforces twice
        it["sources"] = (it["sources"] + [source_id])[-20:]
        if session not in it["sessions"]:
            it["sessions"] = (it["sessions"] + [session])[-20:]
        it["seen"] += 1
        it["pos"] = round(it["pos"] + 0.5, 4)
        if ts and ts > (it["ts_last"] or ""):
            it["ts_last"] = ts
        if it["status"] == "pruned":
            it["status"] = "active"
            self.stats["revived"] += 1
        self.stats["reinforced"] += 1

    # ---------------------------------------------------------------- credit
    def credit(self, session_rewards: dict[str, list[tuple[float, float]]]) -> None:
        """session -> [(event_ts_epoch, reward)]; credit each uncredited retrieval with the mean reward after it."""
        credited = set(self.meta.get("credited", []))
        rows = list(iter_jsonl(self.retr_path))
        counts: dict[str, int] = {}
        for r in rows:
            for iid in r.get("ids", []):
                counts[iid] = counts.get(iid, 0) + 1
        for iid, it in self.items.items():
            it["retrievals"] = counts.get(iid, 0)
        for r in rows:
            rid = r.get("id")
            if not rid or rid in credited:
                continue
            rs = [rw for t, rw in session_rewards.get(r.get("session") or "", []) if t >= parse_ts(r.get("ts")) - 1]
            if not rs:
                continue
            m = sum(rs) / len(rs)
            for iid in r.get("ids", []):
                it = self.items.get(iid)
                if it:
                    self._decay(it, r.get("ts"))
                    it["pos"] = round(it["pos"] + m, 4)
                    it["neg"] = round(it["neg"] + (1 - m), 4)
            credited.add(rid)
            self.stats["credited_retrievals"] += 1
        self.meta["credited"] = sorted(credited)

    def _decay(self, it: dict, ts: str | None) -> None:
        t0, t1 = parse_ts(it.get("decay_ts")), parse_ts(ts) or _now()
        if t0 and t1 > t0:
            f = 0.5 ** ((t1 - t0) / 86400 / HALF_LIFE_DAYS)
            it["pos"], it["neg"] = round(it["pos"] * f, 4), round(it["neg"] * f, 4)
            it["decay_ts"] = datetime.fromtimestamp(t1, timezone.utc).isoformat(timespec="seconds")

    # ---------------------------------------------------------------- forgetting
    def refresh_and_prune(self) -> None:
        now = _now()
        for it in self.items.values():
            pos, neg = it["pos"], it["neg"]
            t0 = parse_ts(it.get("decay_ts"))
            if t0 and now > t0:  # value uses decayed counts without mutating the stored ones (keeps re-runs idempotent)
                f = 0.5 ** ((now - t0) / 86400 / HALF_LIFE_DAYS)
                pos, neg = pos * f, neg * f
            it["value"] = round((1 + pos) / (2 + pos + neg), 4)
            if it["status"] != "active":
                continue
            age_days = (now - parse_ts(it.get("ts_last"))) / 86400 if it.get("ts_last") else 0
            if (it["retrievals"] >= PRUNE_MIN_USES and it["value"] < PRUNE_VALUE) or \
               (it["retrievals"] == 0 and it["seen"] <= 1 and age_days > COLD_DAYS):
                it["status"] = "pruned"
                self.stats["pruned"] += 1
        active = [it for it in self.items.values() if it["status"] == "active"]
        if len(active) > self.cfg.mem_max_active:
            active.sort(key=lambda x: (x["value"], x.get("ts_last") or "", x["id"]))
            for it in active[: len(active) - self.cfg.mem_max_active]:
                it["status"] = "pruned"
                self.stats["pruned"] += 1

    # ---------------------------------------------------------------- recall
    def recall(self, query: str, k: int = 5, session: str | None = None, log: bool = True) -> list[dict]:
        act = [it for it in self.items.values() if it["status"] == "active"]
        mat = None
        if self.vs and self.vs.ensure([query] + [it["text"] for it in act]):
            qv, mat = self.vs.get(query), self.vs.matrix([it["text"] for it in act])
        scored = []
        q = embed_text(query, self.cfg.mem_dim) if mat is None else None
        for i, it in enumerate(act):
            if mat is not None:
                cos = float(mat[i] @ qv)
            else:
                v = embed_text(it["text"], self.cfg.mem_dim)
                cos = sum(a * b for a, b in zip(q, v))
            if cos > 0:
                scored.append((cos * (0.5 + 0.5 * it.get("value", 0.5)), cos, it))
        scored.sort(key=lambda x: (-x[0], x[2]["id"]))
        top = scored[:k]
        if log and top:
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            ids = [it["id"] for _, _, it in top]
            append_jsonl(self.retr_path, [{"id": "retr:" + sha16(ts, session or "", *ids, query), "ts": ts, "session": session, "ids": ids,
                                           "query_hash": sha16(query)}])
            # retrieval counts are derived from this log on the next run (recall never rewrites the ledger: no race with a run)
        return [{"id": it["id"], "tag": it["tag"], "text": it["text"], "value": it.get("value"), "cos": round(c, 4), "score": round(s, 4)}
                for s, c, it in top]

    # ---------------------------------------------------------------- persistence / export
    def save(self) -> None:
        rows = sorted(self.items.values(), key=lambda x: x["seq"])
        atomic_write_text(self.items_path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows))
        atomic_write_json(self.meta_path, self.meta)

    def export_mesh(self) -> None:
        exe = self.cfg.meshstore_exe
        if not exe.exists():
            self.stats["export_error"] = "meshstore.exe missing"
            return
        active = sorted((it for it in self.items.values() if it["status"] == "active"), key=lambda x: x["seq"])
        ids = [it["id"] for it in active]
        old = self.meta.get("exported", [])
        root = self.cfg.meshstore_root
        ns = self.cfg.mem_namespace if self.vs else self.cfg.mem_namespace + "_hash"
        ns_dir = root / "knowledge" / ns
        if self.meta.get("exported_ns") != ns:
            old = []
        if ids[: len(old)] != old:  # something was pruned / revived out of order: rebuild the namespace
            if ns_dir.exists():
                shutil.rmtree(ns_dir)
            old = []
            self.stats["rebuilt"] = True
        if len(ids) == len(old):
            return
        tmp = self.dir / "export_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        texts = [it["text"] for it in active]
        (tmp / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        mat = self.vs.matrix(texts) if self.vs and self.vs.ensure(texts) else None
        dim = 384 if mat is not None else self.cfg.mem_dim
        with open(tmp / "vecs.f32", "wb") as f:
            if mat is not None:
                f.write(mat.astype("<f4").tobytes())
            else:
                for t in texts:
                    f.write(struct.pack(f"<{dim}f", *embed_text(t, dim)))
        root.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(exe), "put-corpus", str(root), ns, str(tmp / "texts.json"), str(tmp / "vecs.f32"),
                            str(dim), "lifelong"], capture_output=True, text=True, timeout=600)
        shutil.rmtree(tmp, ignore_errors=True)
        if r.returncode != 0:
            self.stats["export_error"] = f"put-corpus exit {r.returncode}"
            return
        try:
            self.stats["exported"] = json.loads(r.stdout.strip().splitlines()[-1]).get("added", 0)
        except (ValueError, IndexError):
            pass
        self.meta["exported"] = ids
        self.meta["exported_ns"] = ns
