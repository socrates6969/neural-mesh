"""bench/tests/test_run_fixes.py - run.py: sandbox-INFRA units are re-scored from the stored answer (no second model call);
HELD/CONF refuse models / think budget / sandbox image that differ from PREREG.json. Offline, no model calls."""
import json, pathlib, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import run  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731
CODE_ITEM = dict(id="c1", split="tune", cat="code", sub="func", prompt="write f", kind="code", answer=None, fn="f", tests=[[1], [2]],
                 expected=["2", "4"])
NUM_ITEM = dict(id="m1", split="tune", cat="math", sub="exact", prompt="1+1?", answer=2, kind="num", tol=0)


@pytest.fixture
def code_run(tmp_path, monkeypatch):
    items = [CODE_ITEM, NUM_ITEM]
    bp = tmp_path / "battery_tune.json"
    bp.write_text(json.dumps(dict(header=dict(split="tune", items_sha=run.json_sha(items)), items=items)), encoding="utf8")
    monkeypatch.setattr(run, "battery_path", lambda s: bp)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "stale_check", lambda *a: None)
    monkeypatch.setattr(run, "claude_cli_version", lambda: "test")
    monkeypatch.setattr(run, "ollama_digest", lambda m=None: "sha256:test")
    monkeypatch.setattr(run, "frontend_prompt", lambda p: (p, {"retrieved": False}))
    for k in run.ENV_KEYS: monkeypatch.delenv(k, raising=False)
    return tmp_path


def _callers(calls, cls="OK"):
    def claude(prompt):
        calls.append(prompt)
        text = "```python\ndef f(x): return 2 * x\n```" if prompt == "write f" else "ANSWER: 2"
        return dict(cls=cls, text=text, wall=0.01, tokens=dict(total=5), cost_usd=0.0, err=None if cls == "OK" else "stop_reason=max_tokens")
    return dict(claude=claude, ollama=claude)


def _rows(tmp_path, tag="r"):
    return [json.loads(l) for l in open(tmp_path / "results" / tag / "raw.jsonl", encoding="utf8")]


def test_sandbox_infra_rescores_stored_answer(code_run, monkeypatch):
    tmp_path = code_run
    args = ["--split", "tune", "--cond", "A,B", "--tag", "r", "--k", "1", "--sandbox", "job", "--max-tries", "2", "--no-analyze"]
    sbx = dict(status="infra", why="daemon down")
    monkeypatch.setattr(run.sandbox, "run_code", lambda *a, **k: dict(sbx))
    calls = []
    assert run.main(args, callers=_callers(calls), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    assert len(calls) == 4                                                      # 2 items x A,B, once each
    bad = [r for r in _rows(tmp_path) if r["status"] == "infra_fail"]
    assert len(bad) == 2 and all(r["sandbox_infra"] and r["model_cls"] == "OK" and r["text"] for r in bad)
    pend = run.pending_rescore(tmp_path / "results" / "r" / "raw.jsonl", "tune")
    assert set(pend) == {("c1", "A", 0), ("c1", "B", 0)}
    calls.clear()
    assert run.main(args, callers=_callers(calls), sleep=NOSLEEP) == run.EXIT_INCOMPLETE   # sandbox still down: still no model call
    assert calls == []
    sbx.update(status="pass", bad=0, n=2, why=None, secs=0.1)                   # sandbox back
    assert run.main(args, callers=_callers(calls), sleep=NOSLEEP) == run.EXIT_OK
    assert calls == []                                                           # the model was never asked again
    final = [r for r in _rows(tmp_path) if r["status"] == "ok" and r["id"] == "c1"]
    assert len(final) == 2 and all(r["correct"] and r["rescored"] == 2 and r["cls"] == "OK" and r["err"] is None for r in final)
    assert run.pending_rescore(tmp_path / "results" / "r" / "raw.jsonl", "tune") == {}
    assert run.main(args, callers=_callers(calls), sleep=NOSLEEP) == run.EXIT_OK and calls == []


def test_rescore_keeps_model_failure(code_run, monkeypatch):
    """a truncated (MODEL) answer whose code hit a sandbox outage is re-scored as model_fail, with its model error restored"""
    tmp_path = code_run
    args = ["--split", "tune", "--cond", "A", "--tag", "r", "--k", "1", "--sandbox", "job", "--max-tries", "1", "--cats", "code", "--no-analyze"]
    monkeypatch.setattr(run.sandbox, "run_code", lambda *a, **k: dict(status="infra", why="down"))
    calls = []
    assert run.main(args, callers=_callers(calls, cls="MODEL"), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    monkeypatch.setattr(run.sandbox, "run_code", lambda *a, **k: dict(status="pass", bad=0, n=2, why=None, secs=0.1))
    assert run.main(args, callers=_callers(calls, cls="MODEL"), sleep=NOSLEEP) == run.EXIT_OK and len(calls) == 1
    r = _rows(tmp_path)[-1]
    assert (r["status"], r["correct"], r["cls"], r["err"]) == ("model_fail", False, "MODEL", "stop_reason=max_tokens")
# ---------------------------------------------------------------- PREREG pins for HELD/CONF
def _prereg_env(monkeypatch):
    pre, pre_sha = run.load_prereg()
    monkeypatch.setenv("MESH_V2", "1"); monkeypatch.delenv("MESH_THRESH", raising=False); monkeypatch.delenv("MESH_FRONTEND", raising=False)
    monkeypatch.delenv("BENCH_THINK", raising=False)
    return pre, pre_sha


DRY = ["--split", "held", "--cond", "A,B,C,D", "--tag", "prereg-pin-test", "--dry-run", "--sandbox", "docker"]


def test_prereg_pins_models_think_image(monkeypatch):
    pre, pre_sha = _prereg_env(monkeypatch)
    assert pre["pins"] == dict(claude_model="haiku", claude_think="2048", local_model="qwen2.5-coder:7b", sandbox_image=run.sandbox.IMAGE)
    assert run.file_sha(BENCH / "sandbox.py") == pre["sandbox_sha"], "bench/sandbox.py changed after pre-registration: update PREREG.json"
    assert run.main(DRY) == run.EXIT_OK
    monkeypatch.setattr(run, "CLAUDE_MODEL", "sonnet")
    assert run.main(DRY) == run.EXIT_PREREG
    monkeypatch.setattr(run, "CLAUDE_MODEL", "haiku")
    monkeypatch.setenv("BENCH_THINK", "8000")
    assert run.main(DRY) == run.EXIT_PREREG
    monkeypatch.setenv("BENCH_THINK", "2048")
    assert run.main(DRY) == run.EXIT_OK                                         # explicit default is the same experiment
    monkeypatch.setattr(run, "LOCAL_MODEL", "qwen2.5-coder:14b")
    assert run.main(DRY) == run.EXIT_PREREG
    assert run.main(DRY[:3] + ["A,B"] + DRY[4:]) == run.EXIT_OK                 # the local model only matters when C/D run
    monkeypatch.setattr(run, "LOCAL_MODEL", "qwen2.5-coder:7b")
    monkeypatch.setattr(run.sandbox, "IMAGE", "python:3.12-slim@sha256:" + "0" * 64)
    assert run.main(DRY) == run.EXIT_PREREG
    monkeypatch.setattr(run.sandbox, "IMAGE", pre["sandbox_image"])
    monkeypatch.setattr(run, "load_prereg", lambda: (dict(pre, sandbox_sha="0" * 64), pre_sha))
    assert run.main(DRY) == run.EXIT_PREREG                                     # sandbox.py (the harness) changed
    no_pins = {k: v for k, v in pre.items() if k != "pins"}
    monkeypatch.setattr(run, "load_prereg", lambda: (no_pins, pre_sha))
    assert run.main(DRY) == run.EXIT_PREREG
    assert run.main(["--split", "tune", "--cond", "A", "--tag", "prereg-pin-test", "--dry-run", "--sandbox", "job"]) == run.EXIT_OK   # TUNE unaffected
    assert not (run.RESULTS / "prereg-pin-test").exists()
