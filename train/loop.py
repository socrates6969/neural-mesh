"""Batch training loop: learn by doing -> write pack -> held-out test -> keep/revert -> feed the router.

python -m train.loop --batch batch1.json --rounds 3 --budget 150
Per agent and round r:
  1. TUNE   run the agent (with its current pack) on a fresh slice of tune tasks; collect failures.
  2. WRITE  the pack writer (web research + failures + reference solutions) proposes a new pack.
  3. HELD   on a fresh, never-seen held-out slice: run the agent WITHOUT pack and WITH the candidate pack.
  4. KEEP   the candidate replaces the current pack only if it did not lose on this held slice (diff >= 0);
            a pack is PROMOTED (eligible for the live mesh) only when the pooled held-out evidence of the kept
            pack lineage is significant: McNemar p < 0.05 and more pack-only than base-only wins.
  5. FEED   every held-out outcome goes to a COPY of the router state (feedback --reward), and to leaderboard.json.
Resumable: every agent run is appended to results.jsonl and skipped on restart.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import check, data, packs, promote, runner, stats

ROOT = Path(os.environ.get("MESHTRAIN_ROOT", r"C:\Users\mariu\ruv\hive\runs\agent-catalog"))
REPO = Path(__file__).resolve().parent.parent
_lock = threading.Lock()


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class Trainer:
    def __init__(self, batch: dict, out: Path, bench: Path, budget: float, workers: int, tune_n: int, held_n: int):
        self.batch, self.out, self.bench = batch, out, bench
        self.quota_hit = False
        self.budget, self.workers, self.tune_n, self.held_n = budget, workers, tune_n, held_n
        self.out.mkdir(parents=True, exist_ok=True)
        self.results = out / "results.jsonl"
        self.done = {}
        if self.results.exists():
            for l in self.results.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    self.done[(r["round"], r["phase"], r["agent"], r["task"])] = r
        self.spent = sum(r.get("cost", 0) for r in self.done.values()) + self._writer_spend()
        cat = json.loads((ROOT / "agent_catalog.json").read_text(encoding="utf-8"))["agents"]
        self.catalog = {r["name"]: r for r in cat}
        self.tasks = {f: data.load(bench, f) for f in {a["field"] for a in batch["agents"]}}

    # ---------- bookkeeping ----------
    def _writer_spend(self) -> float:
        p = self.out / "writer_costs.jsonl"
        return sum(json.loads(l)["cost"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.exists() else 0.0

    def log(self, msg: str):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        with _lock, open(self.out / "train.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def pack_dir(self, agent: str, field: str) -> Path:
        d = self.out / "packs" / slug(agent) / field
        d.mkdir(parents=True, exist_ok=True)
        return d

    def state(self, agent: str, field: str) -> dict:
        p = self.pack_dir(agent, field) / "status.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"current": "", "version": 0, "rounds": [], "status": "none"}

    def save_state(self, agent: str, field: str, st: dict):
        d = self.pack_dir(agent, field)
        (d / "status.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
        if st.get("current"):
            (d / "current.md").write_text(st["current"], encoding="utf-8")

    # ---------- running ----------
    def run_many(self, jobs: list[tuple]) -> list[dict]:
        """jobs: (round, phase, agent_spec, task, pack). Returns records in job order (cached ones reused)."""
        out = [None] * len(jobs)
        todo = []
        for i, (rnd, phase, ag, t, pk) in enumerate(jobs):
            k = (rnd, phase, ag["name"], t["id"])
            if k in self.done:
                out[i] = self.done[k]
            else:
                todo.append(i)

        def work(i):
            rnd, phase, ag, t, pk = jobs[i]
            if self.spent >= self.budget or self.quota_hit:
                return i, None
            rec = runner.run(ag, t, pk, self.out / "work")
            if (rec["error"] or "").startswith("quota"):
                with _lock:
                    if not self.quota_hit:
                        self.log("QUOTA: plan usage limit reached (%s) - stopping; nothing recorded" % rec["error"][:120])
                    self.quota_hit = True
                return i, None
            rec.update(round=rnd, phase=phase, passed=bool(check.grade(t, rec["text"])) if not rec["error"] else False)
            rec["text"] = rec["text"][-4000:]
            with _lock:
                self.spent += rec["cost"]
                self.done[(rnd, phase, ag["name"], t["id"])] = rec
                with open(self.results, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return i, rec

        with cf.ThreadPoolExecutor(self.workers) as ex:
            for i, rec in ex.map(work, todo):
                out[i] = rec
        return out

    def slices(self, field: str, rnd: int) -> tuple[list[dict], list[dict]]:
        tune = [t for t in self.tasks[field] if t["split"] == "tune"]
        held = [t for t in self.tasks[field] if t["split"] == "held"]
        s = (rnd * self.tune_n) % max(1, len(tune))
        tune_slice = (tune[s:] + tune[:s])[: self.tune_n]
        held_slice = held[rnd * self.held_n:(rnd + 1) * self.held_n]
        return tune_slice, held_slice

    # ---------- one round for one agent ----------
    def round_for(self, spec: dict, rnd: int) -> dict | None:
        name, field = spec["agent"], spec["field"]
        ag = runner.load_agent(self.catalog[name])
        st = self.state(name, field)
        if any(r["round"] == rnd for r in st["rounds"]):
            return None
        tune, held = self.slices(field, rnd)
        if len(held) < self.held_n:
            self.log("%s/%s: held-out pool exhausted at round %d" % (name, field, rnd))
            return None
        cur = st["current"] or None
        trec = self.run_many([(rnd, "tune", ag, t, cur) for t in tune])
        if any(r is None for r in trec):
            return None
        fails = [{"task": t, "rec": r} for t, r in zip(tune, trec) if not r["passed"]]
        n_ok = sum(r["passed"] for r in trec)
        self.log("%s/%s r%d tune %d/%d (pack v%d)" % (name, field, rnd, n_ok, len(tune), st["version"]))
        cand_path = self.pack_dir(name, field) / ("candidate_r%d.md" % rnd)
        if cand_path.exists():
            cand = cand_path.read_text(encoding="utf-8")
        else:
            cand, wcost = packs.write_pack(ag, field, st["current"], fails, n_ok)
            if runner.is_quota(cand):
                self.quota_hit = True
                self.log("QUOTA: pack writer hit the plan usage limit - stopping")
                return None
            with _lock:
                self.spent += wcost
                with open(self.out / "writer_costs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"agent": name, "field": field, "round": rnd, "cost": wcost}) + "\n")
            leak = packs.leaks(cand, [t for t in self.tasks[field] if t["split"] == "tune"])
            if leak:
                self.log("%s/%s r%d candidate quotes tune tasks %s -> discarded" % (name, field, rnd, leak[:3]))
                cand = ""
            cand_path.write_text(cand, encoding="utf-8")
        if not cand.strip():
            st["rounds"].append({"round": rnd, "tune": [n_ok, len(tune)], "note": "no candidate"})
            self.save_state(name, field, st)
            return None
        base = self.run_many([(rnd, "held_base", ag, t, None) for t in held])
        withp = self.run_many([(rnd, "held_pack", ag, t, cand) for t in held])
        if any(r is None for r in base + withp):
            return None
        cmp_ = stats.paired([r["passed"] for r in base], [r["passed"] for r in withp])
        kept = cmp_["diff"] >= 0
        if kept:
            st["current"], st["version"] = cand, st["version"] + 1
        st["rounds"].append({"round": rnd, "tune": [n_ok, len(tune)], "held": cmp_, "kept": kept, "version": st["version"]})
        pooled = self.pooled(st)
        st["pooled"] = pooled
        st["status"] = ("promoted" if pooled and pooled["p"] < 0.05 and pooled["c_pack_only"] > pooled["b_base_only"]
                        else "candidate" if st["version"] else "none")
        self.save_state(name, field, st)
        if st["status"] == "promoted":
            self.log("%s/%s live registry: %s" % (name, field, promote.publish(name, field, st)))
        self.feed_router(field, held, base, withp)
        self.log("%s/%s r%d held base %d/%d pack %d/%d p=%.3f -> %s (status %s)" % (
            name, field, rnd, cmp_["base"], cmp_["n"], cmp_["pack"], cmp_["n"], cmp_["p"], "KEPT" if kept else "reverted", st["status"]))
        return cmp_

    def pooled(self, st: dict) -> dict | None:
        b = c = n = kb = kp = 0
        for r in st["rounds"]:
            h = r.get("held")
            if h and r.get("kept"):
                b, c, n, kb, kp = b + h["b_base_only"], c + h["c_pack_only"], n + h["n"], kb + h["base"], kp + h["pack"]
        if not n:
            return None
        return {"n": n, "base": kb, "pack": kp, "b_base_only": b, "c_pack_only": c, "p": stats.mcnemar_exact(b, c)}

    # ---------- router + leaderboard ----------
    def feed_router(self, field: str, held: list[dict], base: list[dict], withp: list[dict]):
        exe = REPO / "target" / "release" / "neural-mesh.exe"
        if not exe.exists():
            exe = Path(r"C:\Users\mariu\neural-mesh\target\release\neural-mesh.exe")
        state = self.out / "router_state.trained.json"
        with _lock:
            if not state.exists():
                shutil.copyfile(r"C:\Users\mariu\neural-mesh\data\lifelong\mesh\router_state.json", state)
            try:
                from ingest.extract import features
            except Exception:
                features = lambda s: re.findall(r"[a-z]{4,}", s.lower())[:12]
            for t, rb, rp in zip(held, base, withp):
                feats = ["field_" + field.replace("-", "_")] + [f for f in features(t["prompt"][:600]) if "," not in f][:15]
                for rec in (rb, rp):
                    subprocess.run([str(exe), "feedback", "--state", str(state), "--features", ",".join(feats),
                                    "--agent", rec["agent"], "--reward", "1" if rec["passed"] else "0"],
                                   capture_output=True, timeout=30)

    def export_live(self):
        """Feed verified results into the live mesh router + nightly LLM data (idempotent; see train/export_mesh.py)."""
        try:
            r = subprocess.run([sys.executable, "-m", "train.export_mesh", self.out.name], capture_output=True, text=True,
                               cwd=str(REPO), timeout=3600, env={**os.environ, "PYTHONPATH": str(REPO)})
            self.log("live mesh export: " + " | ".join(x for x in r.stdout.splitlines() if x.startswith(("router", "llm"))) or r.stderr[-300:])
            from . import experience
            self.log("experience memory: +%d verified solutions" % experience.build_from_results(self.out.name))
        except Exception as e:
            self.log("live mesh export failed: %r" % e)

    def leaderboard(self) -> dict:
        board = {}
        for spec in self.batch["agents"]:
            st = self.state(spec["agent"], spec["field"])
            rows = [r for r in self.done.values() if r["agent"] == spec["agent"] and r["field"] == spec["field"]]
            hb = [r for r in rows if r["phase"] == "held_base"]
            hp = [r for r in rows if r["phase"] == "held_pack"]
            board.setdefault(spec["field"], []).append({
                "agent": spec["agent"], "held_base": [sum(r["passed"] for r in hb), len(hb)],
                "held_with_pack": [sum(r["passed"] for r in hp), len(hp)], "pack_version": st["version"],
                "pack_status": st["status"], "pooled": st.get("pooled")})
        for f in board:
            board[f].sort(key=lambda x: -(max(x["held_base"][0] / max(1, x["held_base"][1]),
                                                 x["held_with_pack"][0] / max(1, x["held_with_pack"][1]) if x["pack_status"] != "none" else 0)))
        (self.out / "leaderboard.json").write_text(json.dumps({"spent_usd": round(self.spent, 2), "fields": board}, indent=1), encoding="utf-8")
        return board

    def train(self, rounds: int, start: int = 0):
        self.log("start: %d agents, rounds %d..%d, budget $%.0f, spent so far $%.2f" % (len(self.batch["agents"]), start, start + rounds - 1, self.budget, self.spent))
        for rnd in range(start, start + rounds):
            with cf.ThreadPoolExecutor(min(len(self.batch["agents"]), max(1, self.workers // 2))) as ex:
                list(ex.map(lambda s: self.round_for(s, rnd), self.batch["agents"]))
            self.leaderboard()
            self.log("round %d done, spent $%.2f" % (rnd, self.spent))
            self.export_live()
            if self.spent >= self.budget:
                self.log("budget reached - stopping (resume later with a higher --budget)")
                break
            if self.quota_hit:
                self.log("stopped on plan usage limit - rerun the same command after the limit resets (resumes)")
                break
        self.leaderboard()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--budget", type=float, default=150.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tune-n", type=int, default=16)
    p.add_argument("--held-n", type=int, default=20)
    a = p.parse_args(argv)
    batch = json.loads(Path(a.batch).read_text(encoding="utf-8-sig"))
    out = ROOT / "train" / batch["name"]
    Trainer(batch, out, ROOT / "bench", a.budget, a.workers, a.tune_n, a.held_n).train(a.rounds, a.start)


if __name__ == "__main__":
    main()
