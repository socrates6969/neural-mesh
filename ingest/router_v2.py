"""Router v2 (hive router-v2, user-approved 2026-10-22): MiniLM task embeddings + a contextual bandit, behind the same
`route` / `feedback` CLI contract as neural-mesh.exe (so meshio, export_mesh and route_shadow keep working; opt-in via MESH_ROUTER=v2).

Task features: all-MiniLM-L6-v2 384-d L2-normalised embedding of the task text (llm/vectors.py, onnxruntime) plus two categorical
features, the project and the benchmark field tag (``field_*`` feature) when present.

Learner: one Bayesian linear regression of reward per agent (disjoint LinUCB arms), solved in kernel (dual) form, which is exact for a
linear kernel and keeps the state small (observations, not d x d matrices):
    k(x, x') = <e, e'> + w_p [proj = proj'] + w_f [field = field' != ""] + w_b
    mu_a(x) = m0 + k_a(x)' (K_a + lam I)^-1 (y_a - m0),   sd_a(x)^2 = k(x, x) - k_a(x)' (K_a + lam I)^-1 k_a(x)
Prior mean m0 = 0: an agent with no evidence has no predicted success. The bias term w_b learns each agent's base rate.

Exploration (capped optimism): score_a = mu_a + beta * min(sd_a, kappa * sd_min), sd_min = the smallest sd among the CALLER's candidates.
An untried agent (sd = prior sd, about 1.5) therefore gets at most kappa times the bonus of the best-evidenced candidate, so it can only
win when its predicted success is within beta * (kappa - 1) * sd_min of that agent's, i.e. when the uncertainties are comparable, never
on uncertainty alone. Optimism is computed only over the candidates passed in; agents outside the set are never scored.

Feed (``feedback``): ``--label X`` (a successful dispatch) sets reward 1 for X and reward 0 for the router's current greedy pick when it
differs (the leader's choice is the label, as in the v1 label feed); ``--agent X --reward r`` records r for X; ``--failure-class env``
updates no skill estimate (only a per-agent env-failure count). Continuations are not routing decisions and are not fed by meshio.

CLI: python -m ingest.router_v2 route    --state F [--text T] --features a,b [--candidates x,y] [--project p] [--field f]
     python -m ingest.router_v2 feedback --state F [--text T] --features a,b (--label X | --agent X --reward r [--failure-class env])
     python -m ingest.router_v2 stats    --state F
     python -m ingest.router_v2 rebuild  --outcomes outcomes.jsonl --out router_state.v2.json
v1-only flags (--seed --prior --prior-k --team --type --store --world --eta ...) are accepted and ignored.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

DIM = 384
VERSION = "router-v2/1"
ENV_SIGNALS = {"turn_limit", "redispatch_cue", "api_error"}


@dataclass
class Config:
    lam: float = 1.0          # observation noise / ridge (in prior-variance units)
    w_p: float = 0.5          # project kernel weight
    w_f: float = 0.5          # field kernel weight
    w_b: float = 0.25         # per-agent bias (base rate) kernel weight
    beta: float = 0.05        # UCB width
    kappa: float = 2.0        # bonus cap, as a multiple of the smallest candidate sd
    m0: float = 0.0           # prior mean predicted success
    cap: int = 1500           # max observations kept per agent (oldest dropped)


# ---- BLAS threads --------------------------------------------------------------------------------------------------------------------------

class blas_threads:
    """Run the small linear algebra here with few OpenBLAS threads, then restore the process setting. On this machine the default thread
    count made a 573x573 inverse take ~3 s instead of ~40 ms (oversubscription). Best effort: a no-op when the symbols are not found."""

    def __init__(self, n: int | None = None):
        self.n = int(n or os.environ.get("ROUTER_V2_BLAS_THREADS", "1"))
        self.lib = self.prev = None

    def __enter__(self):
        try:
            import ctypes
            import glob
            d = Path(np.__file__).resolve().parent.parent / "numpy.libs"
            dll = next(iter(glob.glob(str(d / "*openblas*.dll")) + glob.glob(str(d / "*openblas*.so*"))), None)
            if dll:
                lib = ctypes.CDLL(dll)
                get = getattr(lib, "scipy_openblas_get_num_threads64_", None)
                put = getattr(lib, "scipy_openblas_set_num_threads64_", None)
                if get and put:
                    self.lib, self.prev = (get, put), int(get())
                    put(self.n)
        except Exception:
            self.lib = None
        return self

    def __exit__(self, *exc):
        if self.lib and self.prev:
            try:
                self.lib[1](self.prev)
            except Exception:
                pass


# ---- embeddings ----------------------------------------------------------------------------------------------------------------------------

_ENC = None


def _fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic hashed bag-of-words stand-in (ROUTER_V2_FAKE_EMBED=1, tests only)."""
    import zlib
    out = np.zeros((len(texts), DIM), np.float32)
    for i, t in enumerate(texts):
        for w in str(t).lower().split():
            out[i, zlib.crc32(w.encode()) % DIM] += 1.0
        out[i, DIM - 1] += 1e-3
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def embed(texts: list[str]) -> np.ndarray:
    """MiniLM 384-d L2-normalised vectors, one row per text (model loaded once per process)."""
    global _ENC
    if os.environ.get("ROUTER_V2_FAKE_EMBED") == "1":
        return _fake_embed(texts)
    if _ENC is None:
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from llm.vectors import Encoder
        _ENC = Encoder()
    return _ENC.encode([t or " " for t in texts])


def field_of(features: list[str] | None, field: str | None = None) -> str:
    if field:
        return str(field)
    for f in features or []:
        if str(f).startswith("field_"):
            return str(f)[6:]
    return ""


def task_text(text: str | None, features: list[str] | None) -> str:
    """The text that is embedded: the task text when the caller has it, else the keyword bag (degraded, v1 callers)."""
    return text if text and text.strip() else " ".join(f for f in (features or []) if f != "_none")


# ---- the bandit ----------------------------------------------------------------------------------------------------------------------------

class Arm:
    """Observations of one agent with (K + lam I)^-1 kept up to date incrementally (block inverse, O(n^2) per added observation)."""

    def __init__(self):
        self.E = np.zeros((0, DIM), np.float64)
        self.P: list[str] = []
        self.F: list[str] = []
        self.y: list[float] = []
        self._p = np.zeros(0, object)
        self._f = np.zeros(0, object)
        self._inv = None
        self._alpha = None
        self._cfg = None

    def kvec(self, e: np.ndarray, p: str, f: str, c: Config) -> np.ndarray:
        k = self.E @ np.asarray(e, np.float64) + c.w_b
        if c.w_p:
            k += c.w_p * (self._p == p)
        if c.w_f and f:
            k += c.w_f * (self._f == f)
        return k

    @staticmethod
    def kself(f: str, c: Config) -> float:
        return 1.0 + c.w_b + c.w_p + (c.w_f if f else 0.0)

    def add(self, e: np.ndarray, p: str, f: str, y: float, c: Config) -> None:
        e = np.asarray(e, np.float64)
        if self._inv is not None and self._cfg == c and len(self.y) < c.cap and (len(self.y) + 1) % 256:     # periodic full refit bounds drift
            b = self.kvec(e, p, f, c)
            u = self._inv @ b
            s = self.kself(f, c) + c.lam - float(b @ u)
            n = len(self.y)
            inv = np.empty((n + 1, n + 1))
            inv[:n, :n] = self._inv + np.outer(u, u) / s
            inv[:n, n] = inv[n, :n] = -u / s
            inv[n, n] = 1.0 / s
            self._inv = inv
        else:
            self._inv = None
        self.E = np.vstack([self.E, e[None, :]])[-c.cap:]
        self.P, self.F, self.y = (self.P + [p])[-c.cap:], (self.F + [f])[-c.cap:], (self.y + [float(y)])[-c.cap:]
        self._p, self._f = np.array(self.P, object), np.array(self.F, object)
        self._alpha = None

    def _fit(self, c: Config) -> None:
        if self._inv is None or self._cfg != c:
            K = self.E @ self.E.T + c.w_b
            if c.w_p:
                K += c.w_p * (self._p[:, None] == self._p[None, :])
            if c.w_f:
                K += c.w_f * ((self._f[:, None] == self._f[None, :]) & (self._f[:, None] != ""))
            self._inv = np.linalg.inv(K + c.lam * np.eye(len(self.y)))
            self._cfg, self._alpha = c, None
        if self._alpha is None:
            self._alpha = self._inv @ (np.asarray(self.y) - c.m0)

    def predict(self, e: np.ndarray, p: str, f: str, c: Config) -> tuple[float, float]:
        kxx = self.kself(f, c)
        if not self.y:
            return c.m0, float(np.sqrt(kxx))
        self._fit(c)
        k = self.kvec(e, p, f, c)
        var = kxx - float(k @ self._inv @ k)
        return c.m0 + float(k @ self._alpha), float(np.sqrt(max(var, 1e-12)))

class Bandit:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.arms: dict[str, Arm] = {}
        self.env_failures: dict[str, int] = {}
        self.feedbacks = 0

    def agents(self) -> list[str]:
        return list(self.arms)

    def arm(self, a: str) -> Arm:
        return self.arms.setdefault(a, Arm())

    def scores(self, e, p: str, f: str, candidates: list[str]) -> dict[str, dict]:
        """Per candidate: mean predicted success, sd, capped UCB score. Unknown candidates use the prior."""
        c = self.cfg
        pred = {a: (self.arms[a].predict(e, p, f, c) if a in self.arms else Arm().predict(e, p, f, c)) for a in candidates}
        sd_min = min(sd for _, sd in pred.values()) if pred else 0.0
        return {a: {"mean": mu, "sd": sd, "ucb": mu + c.beta * min(sd, c.kappa * sd_min)} for a, (mu, sd) in pred.items()}

    def route(self, e, p: str, f: str, candidates: list[str], greedy: bool = False) -> str | None:
        """Highest score; ties broken by more evidence, then candidate order (deterministic)."""
        if not candidates:
            return None
        sc = self.scores(e, p, f, candidates)
        key = "mean" if greedy else "ucb"
        order = {a: i for i, a in enumerate(candidates)}
        return max(candidates, key=lambda a: (round(sc[a][key], 12), len(self.arms[a].y) if a in self.arms else 0, -order[a]))

    def update(self, agent: str, e, p: str, f: str, reward: float) -> None:
        self.arm(agent).add(np.asarray(e), p, f, float(np.clip(reward, 0.0, 1.0)), self.cfg)

    def feedback_label(self, label: str, e, p: str, f: str, routed: str | None = None) -> str | None:
        """The leader's successful choice is the label: label r=1; the router's greedy pick (over all known agents unless `routed` is given) r=0."""
        if routed is None and self.arms:
            routed = self.route(e, p, f, self.agents(), greedy=True)
        if routed and routed != label:
            self.update(routed, e, p, f, 0.0)
        self.update(label, e, p, f, 1.0)
        self.feedbacks += 1
        return routed

    def feedback_reward(self, agent: str, e, p: str, f: str, reward: float) -> None:
        self.update(agent, e, p, f, reward)
        self.feedbacks += 1

    def feedback_env(self, agent: str) -> None:
        self.env_failures[agent] = self.env_failures.get(agent, 0) + 1
        self.feedbacks += 1

    # ---- persistence (JSON; embeddings as base64 float16) ----
    def to_json(self) -> dict:
        def b64(E):
            return base64.b64encode(E.astype(np.float16).tobytes()).decode()
        return {"version": VERSION, "encoder": "all-MiniLM-L6-v2", "config": asdict(self.cfg), "feedbacks": self.feedbacks,
                "env_failures": self.env_failures, "agents": self.agents(),
                "arms": {a: {"E": b64(r.E), "P": r.P, "F": r.F, "y": r.y} for a, r in self.arms.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Bandit":
        b = cls(Config(**{k: v for k, v in (d.get("config") or {}).items() if k in Config.__dataclass_fields__}))
        b.feedbacks = int(d.get("feedbacks", 0))
        b.env_failures = dict(d.get("env_failures") or {})
        for a, r in (d.get("arms") or {}).items():
            arm = b.arm(a)
            arm.E = np.frombuffer(base64.b64decode(r["E"]), np.float16).astype(np.float64).reshape(-1, DIM)
            arm.P, arm.F, arm.y = list(r["P"]), list(r["F"]), [float(x) for x in r["y"]]
            arm._p, arm._f = np.array(arm.P, object), np.array(arm.F, object)
        return b


# ---- state file I/O (atomic write under a lock file; route never writes) ----------------------------------------------------------------

def load(path: Path) -> Bandit:
    path = Path(path)
    if not path.exists():
        return Bandit()
    return Bandit.from_json(json.loads(path.read_text(encoding="utf-8")))


def save(b: Bandit, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(b.to_json()), encoding="utf-8")
    os.replace(tmp, path)


class Lock:
    """Exclusive lock file next to the state; a lock older than `stale` seconds is considered abandoned."""

    def __init__(self, path: Path, timeout: float = 30.0, stale: float = 120.0):
        self.p, self.timeout, self.stale = Path(str(path) + ".lock"), timeout, stale

    def __enter__(self):
        t0 = time.monotonic()
        self.p.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                os.close(os.open(self.p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return self
            except FileExistsError:
                try:
                    if time.time() - self.p.stat().st_mtime > self.stale:
                        self.p.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() - t0 > self.timeout:
                    raise TimeoutError(f"lock {self.p}")
                time.sleep(0.05)

    def __exit__(self, *exc):
        self.p.unlink(missing_ok=True)


def check_not_v1(path: Path) -> None:
    """v2 never writes a v1 state (router_state.json is the live Hebbian router)."""
    p = Path(path)
    if p.name == "router_state.json":
        raise ValueError(f"refusing to write the v1 router state {p}; use router_state.v2.json")
    if p.exists():
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("version") != VERSION:
                raise ValueError(f"{p} is not a {VERSION} state")
        except json.JSONDecodeError as ex:
            raise ValueError(f"{p} is not JSON: {ex}") from None


# ---- outcome-level API (meshio / rebuild) ---------------------------------------------------------------------------------------------------

def is_env_failure(o: dict) -> bool:
    return o.get("outcome") == "re-dispatch" and bool(ENV_SIGNALS & set(o.get("signals") or []))


def agent_of(o: dict) -> str:
    return (o.get("agent") or "general-purpose").replace(",", "_")


def fed_kind(o: dict) -> bool:
    """Routing decisions only: dispatches and grader-verified benchmark solves (continuations go to an agent already chosen)."""
    return o.get("kind") in ("dispatch", "benchmark")


def apply_outcome(b: Bandit, o: dict, e: np.ndarray | None = None) -> str:
    """Feed one outcome row with the v2 policy; returns the update kind."""
    agent = agent_of(o)
    if is_env_failure(o):
        b.feedback_env(agent)
        return "env"
    if e is None:
        e = embed([task_text(o.get("task_text"), o.get("features"))])[0]
    p, f = o.get("project") or "default", field_of(o.get("features"))
    if o.get("outcome") == "success":
        b.feedback_label(agent, e, p, f)
        return "label"
    b.feedback_reward(agent, e, p, f, float(o.get("reward") or 0.0))
    return "reward"


def feed_outcome(state: Path, o: dict) -> str:
    """meshio entry point: one outcome into the v2 state file (read-modify-write under the lock)."""
    if not fed_kind(o):
        return "skipped"
    check_not_v1(state)
    e = None if is_env_failure(o) else embed([task_text(o.get("task_text"), o.get("features"))])[0]
    with Lock(state), blas_threads():
        b = load(state)
        kind = apply_outcome(b, o, e)
        save(b, state)
    return kind


def rebuild(outcomes: Path, out: Path, cfg: Config | None = None) -> dict:
    """Replay outcomes.jsonl (time order, dispatch + benchmark rows) into a NEW v2 state file. Never touches the v1 state."""
    check_not_v1(out)
    rows = [json.loads(l) for l in Path(outcomes).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = sorted((r for r in rows if fed_kind(r)), key=lambda r: (r.get("ts") or "", r.get("id") or ""))
    E = embed([task_text(r.get("task_text"), r.get("features")) for r in rows]) if rows else np.zeros((0, DIM))
    b = Bandit(cfg)
    kinds: dict[str, int] = {}
    with blas_threads():
        for r, e in zip(rows, E):
            k = apply_outcome(b, r, e)
            kinds[k] = kinds.get(k, 0) + 1
    with Lock(out):
        save(b, out)
    return {"ok": True, "out": str(out), "rows": len(rows), "kinds": kinds, "agents": len(b.arms), "config": asdict(b.cfg)}


# ---- CLI ----------------------------------------------------------------------------------------------------------------------------------

def _split(s: str | None) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def _parser():
    import argparse
    p = argparse.ArgumentParser(prog="python -m ingest.router_v2")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("route", "feedback"):
        s = sub.add_parser(name)
        s.add_argument("--state", type=Path, required=True)
        s.add_argument("--features", default="")
        s.add_argument("--text")
        s.add_argument("--project", default="default")
        s.add_argument("--field")
        if name == "route":
            s.add_argument("--candidates")
            s.add_argument("--greedy", action="store_true")
        else:
            s.add_argument("--label")
            s.add_argument("--agent")
            s.add_argument("--reward", type=float)
            s.add_argument("--failure-class", default="skill")
    s = sub.add_parser("stats")
    s.add_argument("--state", type=Path, required=True)
    s = sub.add_parser("rebuild")
    s.add_argument("--outcomes", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    for k in Config.__dataclass_fields__:
        s.add_argument("--" + k.replace("_", "-"), type=float)
    return p


def route_cmd(state: Path, features: list[str], candidates: list[str] | None, text: str | None = None, project: str = "default",
              field: str | None = None, greedy: bool = False) -> dict:
    """Read-only: the state file is never written by route."""
    b = load(state)
    cands = candidates or b.agents()
    if not cands:
        return {"ok": False, "error": "no candidates", "router": "v2"}
    e = embed([task_text(text, features)])[0]
    f = field_of(features, field)
    with blas_threads():
        sc = b.scores(e, project, f, cands)
        pick = b.route(e, project, f, cands, greedy=greedy)
        greedy_pick = b.route(e, project, f, cands, greedy=True)
    top = sorted(sc, key=lambda a: -sc[a]["ucb"])[:5]
    return {"ok": True, "router": "v2", "agent": pick, "greedy": greedy_pick, "n_cand": len(cands),
            "scores": {a: {k: round(v, 4) for k, v in sc[a].items()} for a in top}}


def _feedback_cmd(a) -> dict:
    agent = a.label or a.agent
    if not agent:
        raise ValueError("feedback needs --label or --agent")
    check_not_v1(a.state)
    env = a.failure_class == "env"
    if not a.label and a.reward is None and not env:
        raise ValueError("--agent needs --reward")
    feats = _split(a.features)
    e = None if env else embed([task_text(a.text, feats)])[0]
    f = field_of(feats, a.field)
    routed = None
    with Lock(a.state), blas_threads():
        b = load(a.state)
        if env:
            b.feedback_env(agent)
            kind = "env"
        elif a.label:
            routed = b.feedback_label(a.label, e, a.project, f, routed=a.agent)
            kind = "label"
        else:
            b.feedback_reward(agent, e, a.project, f, a.reward)
            kind = "reward"
        save(b, a.state)
    out = {"ok": True, "router": "v2", "kind": kind, "agent": agent, "feedbacks": b.feedbacks}
    if kind == "label":
        out["routed"] = routed
    return out


def main(argv=None) -> int:
    a, unknown = _parser().parse_known_args(argv)     # v1-only flags are accepted and ignored
    try:
        if a.cmd == "route":
            out = route_cmd(a.state, _split(a.features), _split(a.candidates), a.text, a.project, a.field, a.greedy)
        elif a.cmd == "stats":
            b = load(a.state)
            out = {"ok": True, "router": "v2", "feedbacks": b.feedbacks, "config": asdict(b.cfg), "env_failures": b.env_failures,
                   "agents": {x: {"n": len(r.y), "mean_y": round(float(np.mean(r.y)), 4)} for x, r in b.arms.items()}}
        elif a.cmd == "rebuild":
            cfg = Config(**{k: (int(v) if k == "cap" else v) for k, v in vars(a).items() if k in Config.__dataclass_fields__ and v is not None})
            out = rebuild(a.outcomes, a.out, cfg)
        else:
            out = _feedback_cmd(a)
    except (ValueError, TimeoutError, OSError) as ex:
        out = {"ok": False, "error": str(ex), "router": "v2"}
    if unknown:
        out["ignored"] = unknown
    print(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())