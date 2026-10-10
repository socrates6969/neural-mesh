"""Feed outcomes into neural-mesh's `feedback` CLI state (plus its MeshStore journal, so a new router state can warm-start)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .util import sha16


class MeshFeedback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fed_path: Path = cfg.state / "mesh_fed.txt"
        self.fed: set[str] = set()
        if self.fed_path.exists():
            self.fed = {x.strip() for x in self.fed_path.read_text(encoding="utf-8").splitlines() if x.strip()}
        self.stats = {"fed": 0, "failed": 0, "skipped": 0}

    def available(self) -> bool:
        return self.cfg.mesh_exe.exists()

    def feed(self, outcome: dict) -> None:
        """At-most-once per outcome id: the id is recorded BEFORE the call, so a crash can lose one event but never double-count it."""
        oid = outcome["id"]
        if oid in self.fed:
            self.stats["skipped"] += 1
            return
        feats = [f for f in outcome.get("features") or [] if f and "," not in f][:24] or ["_none"]
        agent = (outcome.get("agent") or "general-purpose").replace(",", "_")
        self.fed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fed_path, "a", encoding="utf-8") as f:
            f.write(oid + "\n")
        self.fed.add(oid)
        self.cfg.router_state.parent.mkdir(parents=True, exist_ok=True)
        world = "".join(c if c.isalnum() or c in "-_:" else "_" for c in (outcome.get("project") or "default"))[:64] or "default"
        cmd = [str(self.cfg.mesh_exe), "feedback", "--state", str(self.cfg.router_state), "--features", ",".join(feats),
               "--agent", agent, "--reward", f"{max(0.0, min(1.0, float(outcome['reward']))):.4f}",
               "--store", str(self.cfg.meshstore_root), "--world", world]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            ok = r.returncode == 0 and '"ok":true' in r.stdout.replace(" ", "")
        except (OSError, subprocess.SubprocessError):
            ok = False
        self.stats["fed" if ok else "failed"] += 1


def feature_key(features: list[str]) -> str:
    return sha16(*features)
