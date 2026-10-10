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
