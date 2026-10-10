"""Paths and knobs. Everything can be overridden with environment variables (used by the tests)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    projects: Path = field(default_factory=lambda: Path(os.environ.get("INGEST_PROJECTS", str(Path.home() / ".claude" / "projects"))))
    data: Path = field(default_factory=lambda: Path(os.environ.get("INGEST_DATA", str(REPO / "data" / "lifelong"))))
    ollama_url: str = field(default_factory=lambda: os.environ.get("INGEST_OLLAMA_URL", "http://localhost:11434"))
    judge_model: str = field(default_factory=lambda: os.environ.get("INGEST_JUDGE_MODEL", "qwen2.5-coder:7b"))
    mesh_exe: Path = field(default_factory=lambda: Path(os.environ.get("INGEST_MESH_EXE", str(REPO / "target" / "release" / "neural-mesh.exe"))))
    meshstore_exe: Path = field(default_factory=lambda: Path(os.environ.get("INGEST_MESHSTORE_EXE", str(REPO / "target" / "release" / "meshstore.exe"))))
    settle_minutes: float = 20.0       # an agent / turn is final once idle this long (or once its parent saw the result)
    judge_budget: int = 300            # max NEW judge calls per run (cached ones are free)
    judge_timeout_s: float = 120.0
    sft_min_judge: float = 0.7
    mem_dim: int = 256
    mem_namespace: str = "lifelong"
    mem_max_active: int = 20000
    mesh_feedback: bool = True
    mesh_feed_mode: str = field(default_factory=lambda: os.environ.get("INGEST_MESH_FEED", "label"))  # "label" (default, R117) | "reward" (legacy)

    # derived paths
    @property
    def state(self) -> Path:
        return self.data / "state"

    @property
    def cursor_file(self) -> Path:
        return self.state / "cursor.json"

    @property
    def files_dir(self) -> Path:
        return self.state / "files"

    @property
    def outcomes(self) -> Path:
        return self.data / "outcomes.jsonl"

    @property
    def sft(self) -> Path:
        return self.data / "sft.jsonl"

    @property
    def judge_cache(self) -> Path:
        return self.data / "judge.jsonl"

    @property
    def runs(self) -> Path:
        return self.data / "runs.jsonl"

    @property
    def memory_dir(self) -> Path:
        return self.data / "memory"

    @property
    def meshstore_root(self) -> Path:
        return self.memory_dir / "meshstore"

    @property
    def router_state(self) -> Path:
        return self.data / "mesh" / "router_state.json"

    @property
    def lock_file(self) -> Path:
        return self.data / ".ingest.lock"
