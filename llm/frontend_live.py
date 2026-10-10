"""Live consumers of the front-end (all DEFAULT OFF; nothing here changes behaviour unless the owner opts in explicitly).

S2 `python -m llm ask [--mode shadow|cli|advisory] [--enabled] [--prompt-file F | stdin] [--forward-cmd CMD] [--local-timeout 20]
                      [--min-conf x --min-sim x] [--log-text]`
    Without --enabled: PURE pass-through (the prompt goes to --forward-cmd on stdin, or to stdout), no model, no log.
    With --enabled: signals (verified cache, MiniLM OOD + retrieval, local answer unless nightly.lock is fresh, guarded tool route when
    models/tool_exemplars.jsonl exists) -> the Rust policy -> by mode:
      shadow   : forward the ORIGINAL prompt (no context); the decision is only logged (`would_be`)
      advisory : forward the prompt plus the assembled untrusted context; other decisions are logged as `would_be`
      cli      : serve a cache / local answer on stdout; tool and forward decisions forward with context (no tool executor)
    ANY error -> forward the original prompt without context. --forward-cmd gets the payload on stdin and its exit code is returned.
    Log: models/frontend_live.jsonl, one line per call, prompt sha256 only (text only with --log-text).

S1 `python -m llm shadow-drain` (offline batch, under the llm lock): every prompt the UserPromptSubmit shadow hook (llm/hook.py)
    queued in models/frontend_shadow.jsonl -> signals, local answer, Rust decision, join to Claude's real answer from the ingestor's
    sft rows (same session, same normalised text, not earlier than the prompt), judge -> models/frontend_shadow_drained.jsonl (no text).
    Unjoined prompts stay pending for JOIN_WAIT_S, then are recorded as unjoined."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import config
from .frontend import (TOOL_EXEMPLARS, WRONG, assemble_context, ask_local, chunk_text, git_head, local_model, norm, parse_ts, rust_decide,
                       verified_cache_hit)

LIVE_LOG = config.MODELS / "frontend_live.jsonl"
CFG_FILE = config.MODELS / "frontend_cfg.json"
SHADOW_LOG = config.MODELS / "frontend_shadow.jsonl"
SHADOW_DRAINED = config.MODELS / "frontend_shadow_drained.jsonl"
SHADOW_STATE = config.MODELS / "frontend_shadow_state.json"
LOCK_FRESH_S = 12 * 3600          # same staleness rule as the llm lock (llm/__main__.py)
JOIN_WAIT_S = 7 * 24 * 3600
TOP_CHUNKS = 8
DEFAULT_CTX_BUDGET = 800
MODES = ("shadow", "cli", "advisory")


def sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def nightly_busy(lock: Path | None = None, now: float | None = None) -> bool:
    """a fresh models/nightly.lock means training / eval owns the GPU: do not start the local model"""
    p = Path(lock or config.LOCK)
    try:
        return (now or time.time()) - p.stat().st_mtime < LOCK_FRESH_S
    except OSError:
        return False


def policy_cfg(min_conf=None, min_sim=None, cfg_file: Path | None = None) -> dict:
    """thresholds from a KEEP eval (models/frontend_cfg.json) if present, else the Rust defaults; flags override. `enabled` is
    always set by the caller's explicit --enabled, never by the file."""
    cfg = {}
    try:
        d = json.loads(Path(cfg_file or CFG_FILE).read_text(encoding="utf-8"))
        if d.get("verdict") == "KEEP":
            cfg = {k: d[k] for k in ("min-conf", "min-sim", "max-local-tokens", "ctx-budget") if k in d}
    except (OSError, ValueError, AttributeError):
        pass
    if min_conf is not None:
        cfg["min-conf"] = min_conf
    if min_sim is not None:
        cfg["min-sim"] = min_sim
    return {"enabled": 1, **cfg}


class LiveBackend:
    """MiniLM vectors, Qwen tokenizer counts (tokenizers only; len/4 fallback), the production Ollama model, the local judge"""

    def __init__(self):
        self._enc = self._tok = None
        self.model = local_model()

    def encode(self, texts):
        if self._enc is None:
            from .vectors import Encoder
            self._enc = Encoder()
        return self._enc.encode([t[:2000] for t in texts])

    def ntok(self, s: str) -> int:
        if self._tok is None:
            try:
                from tokenizers import Tokenizer
                self._tok = Tokenizer.from_file(str(config.base_path() / "tokenizer.json"))
            except Exception:
                self._tok = False
        return len(self._tok.encode(s or "").ids) if self._tok else max(1, len(s or "") // 4)

    def ask(self, prompt: str, timeout: float):
        return ask_local(self.model, prompt, timeout=timeout)

    def grade(self, instruction: str, reference: str, answer: str) -> float:
        from . import judge
        return judge.grade(instruction, reference, answer)


def compute_signals(prompt: str, project, ts: str, bank_rows: list[dict], backend, skip_local: bool, local_timeout: float,
                    head_now: str | None = None) -> dict:
    """{sig (CLI keys), chunks (with the exact counted text), answer, cache_why, local_error}; bank = rows strictly earlier than ts"""
    t = parse_ts(ts)
    bank = [r for r in bank_rows if (bt := parse_ts(r.get("ts"))) is not None and t is not None and bt < t]
    hit, why = verified_cache_hit(prompt, project, ts, bank, head_now=head_now)
    out = {"answer": hit, "cache_why": why, "local_error": None, "chunks": []}
    q = backend.encode([prompt])[0]
    max_sim = None
    if bank:
        iv = backend.encode([r["instruction"] for r in bank])
        texts = [chunk_text(r["response"]) for r in bank]
        rv = backend.encode(texts)
        max_sim = float((iv @ q).max())
        sc = rv @ q
        for k in sorted(range(len(bank)), key=lambda i: -float(sc[i]))[:TOP_CHUNKS]:
            out["chunks"].append({"id": bank[k]["id"], "score": float(sc[k]), "tokens": backend.ntok(texts[k]), "text": texts[k]})
    conf, n = None, 0
    if hit is not None:
        n = backend.ntok(hit)
    elif not skip_local:
        try:
            ans, conf, n = backend.ask(prompt, local_timeout)
            out["answer"] = ans
        except Exception as e:  # Ollama down / timeout: no confidence -> the policy forwards
            out["local_error"] = f"{type(e).__name__}: {e}"[:200]
    sig = {"cache-hit": int(hit is not None), "local-conf": conf, "local-tokens": n, "max-sim": max_sim}
    if TOOL_EXEMPLARS.exists():
        sig["tool-exemplars"] = str(TOOL_EXEMPLARS)
        sig["query-vec"] = json.dumps([round(float(x), 6) for x in q])
    out["sig"] = sig
    return out


def _forward(payload: str, forward_cmd: str | None, out) -> int:
    """the explicit --forward-cmd gets the payload on stdin (its stdout/stderr pass through, its exit code is returned); without it
    the payload is written to stdout"""
    if forward_cmd:
        try:
            out.flush()
        except (AttributeError, OSError, ValueError):
            pass
        r = subprocess.run(forward_cmd, shell=True, input=payload, text=True, encoding="utf-8")
        return r.returncode
    out.write(payload if payload.endswith("\n") else payload + "\n")
    return 0


def _log(rec: dict, path: Path | None = None) -> None:
    p = Path(path or LIVE_LOG)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def ask(prompt: str, mode: str = "shadow", enabled: bool = False, forward_cmd: str | None = None, local_timeout: float = 20.0,
        log_text: bool = False, min_conf=None, min_sim=None, project: str | None = None, backend_factory=LiveBackend, decide=rust_decide,
        rows_loader=None, out=None, log_path: Path | None = None, forward=_forward) -> int:
    out = out or sys.stdout
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if not enabled:   # pure pass-through: no model, no policy, no log
        return forward(prompt, forward_cmd, out)
    t0 = time.monotonic()
    rec = {"ts": now_iso(), "sha256": sha(prompt), "mode": mode, "enabled": True}
    payload, served_answer = prompt, None
    try:
        from . import data
        rows = rows_loader() if rows_loader else data.read_jsonl(config.SFT)
        project = project or Path.cwd().name
        busy = nightly_busy()
        backend = backend_factory()
        s = compute_signals(prompt, project, rec["ts"], rows, backend, skip_local=busy, local_timeout=local_timeout, head_now=git_head())
        cfg = policy_cfg(min_conf, min_sim)
        d = decide(cfg, s["sig"], s["chunks"])
        ctx = assemble_context(d.get("context") or [], s["chunks"], int(cfg.get("ctx-budget", DEFAULT_CTX_BUDGET))) if d.get("ok") else ""
        rec.update(decision=d["decision"], reason=d.get("reason"), rust_ok=bool(d.get("ok")), rust_error=d.get("error"), tool=d.get("tool"),
                   nightly_busy=busy, cache_why=s["cache_why"], local_error=s["local_error"], max_sim=s["sig"]["max-sim"], conf=s["sig"]["local-conf"],
                   ntok=s["sig"]["local-tokens"], ctx_ids=d.get("context") or [], ctx_tokens=int(d.get("context_tokens") or 0) if ctx else 0)
        if mode == "shadow":
            served = "forward"
        elif mode == "advisory":
            served, payload = "forward", (ctx + "\n\n" + prompt) if ctx else prompt
        elif d["decision"] in ("cache", "local") and s["answer"]:
            served, served_answer = d["decision"], s["answer"]
        else:
            served, payload = "forward", (ctx + "\n\n" + prompt) if ctx else prompt
        rec["served"] = served
        rec["would_be"] = d["decision"]
        if log_text:
            rec["prompt"], rec["local_answer"] = prompt, s["answer"]
    except Exception as e:   # fail closed: the original prompt, no context
        payload, served_answer = prompt, None
        rec.update(decision="forward", reason="error", served="forward", error=f"{type(e).__name__}: {e}"[:300])
    rec["latency_ms"] = round((time.monotonic() - t0) * 1000)
    if served_answer is not None:
        rc = 0
        out.write(served_answer if served_answer.endswith("\n") else served_answer + "\n")
    else:
        try:
            rc = forward(payload, forward_cmd, out)
        except Exception as e:
            rec["forward_error"] = f"{type(e).__name__}: {e}"[:300]
            rc = 2
    rec["forward_rc"] = rc if served_answer is None else None
    _log(rec, log_path)
    return rc


# ------------------------------------------------------------------------------------------------------------------------------
# S1 offline drain of the shadow queue
def read_shadow(path: Path | None = None) -> list[dict]:
    out = []
    try:
        with open(Path(path or SHADOW_LOG), encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if isinstance(e, dict) and isinstance(e.get("prompt"), str) and e.get("sha256"):
                    out.append(e)
    except FileNotFoundError:
        pass
    return out


def _key(e: dict) -> str:
    return f"{e.get('ts')}|{e.get('sha256')}|{e.get('session')}"


def join_claude(e: dict, rows: list[dict]) -> dict | None:
    """Claude's real answer to a queued prompt: an ingested sft row of the same session whose normalised instruction equals the
    normalised prompt and that is not earlier than the prompt (earliest such row wins)"""
    t, key = parse_ts(e.get("ts")), norm(e.get("prompt", ""))
    if t is None or not key:
        return None
    c = [r for r in rows if r.get("session") == e.get("session") and norm(r.get("instruction", "")) == key
         and (rt := parse_ts(r.get("ts"))) is not None and rt >= t - 60]
    return min(c, key=lambda r: parse_ts(r["ts"])) if c else None


def shadow_drain(backend_factory=LiveBackend, decide=rust_decide, rows: list[dict] | None = None, now: float | None = None,
                 shadow_path: Path | None = None, drained_path: Path | None = None, state_path: Path | None = None,
                 local_timeout: float = 60.0, log=print) -> dict:
    from . import data
    rows = rows if rows is not None else data.read_jsonl(config.SFT)
    state_p = Path(state_path or SHADOW_STATE)
    try:
        done = set(json.loads(state_p.read_text(encoding="utf-8")).get("done", []))
    except (OSError, ValueError, AttributeError):
        done = set()
    now = now or time.time()
    pending = [e for e in read_shadow(shadow_path) if _key(e) not in done]
    stats = {"pending": len(pending), "drained": 0, "joined": 0, "unjoined_expired": 0, "waiting": 0}
    if not pending:
        log("shadow-drain: nothing pending")
        return stats
    backend = backend_factory()
    cfg = policy_cfg()
    for e in pending:
        claude = join_claude(e, rows)
        age = now - (parse_ts(e.get("ts")) or now)
        if claude is None and age < JOIN_WAIT_S:
            stats["waiting"] += 1
            continue
        try:
            s = compute_signals(e["prompt"], e.get("project"), e["ts"], rows, backend, skip_local=False, local_timeout=local_timeout)
            d = decide(cfg, s["sig"], s["chunks"])
            score = backend.grade(e["prompt"], claude["response"], s["answer"]) if (claude and s["answer"]) else None
            rec = {"ts": e["ts"], "sha256": e["sha256"], "session": e.get("session"), "project": e.get("project"), "would_be": d["decision"],
                   "reason": d.get("reason"), "rust_ok": bool(d.get("ok")), "cache_why": s["cache_why"], "max_sim": s["sig"]["max-sim"],
                   "conf": s["sig"]["local-conf"], "ntok": s["sig"]["local-tokens"], "local_error": s["local_error"],
                   "claude_id": claude["id"] if claude else None, "score": score,
                   "wrong_if_served": (score < WRONG) if score is not None else None, "drained_at": now_iso()}
        except Exception as ex:
            rec = {"ts": e["ts"], "sha256": e["sha256"], "session": e.get("session"), "error": f"{type(ex).__name__}: {ex}"[:300], "drained_at": now_iso()}
        _log(rec, Path(drained_path or SHADOW_DRAINED))
        done.add(_key(e))
        stats["drained"] += 1
        stats["joined" if claude else "unjoined_expired"] += 1
    tmp = state_p.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps({"done": sorted(done)}), encoding="utf-8")
    os.replace(tmp, state_p)
    log(f"shadow-drain: {stats}")
    return stats


# ------------------------------------------------------------------------------------------------------------------------------
def cli_ask(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm ask", description="front-end wrapper (S2); pass-through unless --enabled")
    ap.add_argument("--mode", choices=MODES, default="shadow")
    ap.add_argument("--enabled", action="store_true", help="opt in (default OFF = pure pass-through)")
    ap.add_argument("--prompt-file")
    ap.add_argument("--forward-cmd", help="command that receives the (possibly context-augmented) prompt on stdin; its exit code is returned")
    ap.add_argument("--local-timeout", type=float, default=20.0)
    ap.add_argument("--min-conf", type=float)
    ap.add_argument("--min-sim", type=float)
    ap.add_argument("--project")
    ap.add_argument("--log-text", action="store_true", help="also log prompt and local answer text (default: sha256 only)")
    a = ap.parse_args(argv)
    if a.prompt_file:
        prompt = Path(a.prompt_file).read_text(encoding="utf-8")
    else:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        prompt = sys.stdin.read()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return ask(prompt, mode=a.mode, enabled=a.enabled, forward_cmd=a.forward_cmd, local_timeout=a.local_timeout, log_text=a.log_text,
               min_conf=a.min_conf, min_sim=a.min_sim, project=a.project)
