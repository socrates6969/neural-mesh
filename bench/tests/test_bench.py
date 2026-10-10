"""bench/tests/test_bench.py - offline tests for the bench battery (no model calls, no network except a local docker daemon).
Run: python -m pytest bench/tests -q"""
import json, os, pathlib, random, sys, time
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import sandbox  # noqa: E402

DOCKER_OK, DOCKER_WHY = sandbox.docker_available()
needs_docker = pytest.mark.skipif(not DOCKER_OK, reason=f"docker sandbox unavailable: {DOCKER_WHY}")

ESCAPES = {   # candidate -> the value it returns when it is CONTAINED
    "file_write": ("def f():\n    try:\n        open('/etc/bench_pwned', 'w').write('x')\n        open('pwned.txt', 'w').write('x')\n        return 'wrote'\n"
                   "    except Exception:\n        return 'blocked'", "blocked"),
    "socket": ("def f():\n    import socket\n    try:\n        socket.create_connection(('1.1.1.1', 53), 2)\n        return 'connected'\n"
               "    except Exception:\n        return 'blocked'", "blocked"),
    "fork_bomb": ("def f():\n    import os, time\n    n = 0\n    try:\n        while n < 5000:\n            if os.fork() == 0:\n                time.sleep(60); os._exit(0)\n"
                  "            n += 1\n    except OSError:\n        return 'limited'\n    return 'unlimited'", "limited"),
    "infinite_loop": ("def f():\n    while True:\n        pass", None),
}


@needs_docker
@pytest.mark.parametrize("name", list(ESCAPES))
def test_sandbox_contains(name):
    code, contained = ESCAPES[name]
    t0 = time.time()
    r = sandbox.run_code(code, "f", [[]], [sandbox.canon(contained)], mode="docker")
    assert time.time() - t0 < 25, r
    if name == "infinite_loop":
        assert r["status"] == "timeout", r
    else:
        assert r["status"] == "pass", r          # pass == the candidate observed it was contained
    assert not (BENCH / "pwned.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="job-object fallback is Windows-specific")
def test_sandbox_job_fallback_basic():
    """the TUNE-only fallback: right/wrong answers, an audit-hook refusal and the host timeout (NOT a security boundary)"""
    assert sandbox.run_code("def f(x): return x * 2", "f", [[3]], [sandbox.canon(6)], mode="job")["status"] == "pass"
    assert sandbox.run_code("def f(x): return x", "f", [[3]], [sandbox.canon(6)], mode="job")["status"] == "fail"
    code, contained = ESCAPES["file_write"]
    assert sandbox.run_code(code, "f", [[]], [sandbox.canon(contained)], mode="job")["status"] == "pass"
    assert sandbox.run_code(ESCAPES["infinite_loop"][0], "f", [[]], [sandbox.canon(None)], mode="job", timeout=3)["status"] == "timeout"


# ====================================================================== run.py
import run  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731


def _cj(text="ANSWER: 4", stop="end_turn", is_error=False):
    return json.dumps(dict(type="result", result=text, is_error=is_error, stop_reason=stop, usage=dict(input_tokens=10, output_tokens=5),
                           duration_api_ms=100, total_cost_usd=0.001, modelUsage={"claude-haiku-test": {}}))


def _const(cls, text="ANSWER: 4"):
    return lambda prompt: dict(cls=cls, text=text, wall=0.1, tokens=dict(total=1), err=None if cls == "OK" else "x")


def test_classify(monkeypatch):
    c = run.classify_claude
    assert c(False, _cj())[0] == "OK"
    assert c(True, "")[0] == "INFRA"                                          # host timeout
    assert c(False, "Error: socket hang up")[0] == "INFRA"                    # not JSON
    assert c(False, _cj("API Error: 529 Overloaded", is_error=True))[0] == "INFRA"
    assert c(False, _cj("Claude AI usage limit reached", is_error=True))[0] == "INFRA"
    assert c(False, _cj("", is_error=True))[0] == "INFRA"
    assert c(False, _cj("I cannot help with that", is_error=True))[0] == "MODEL"
    assert c(False, _cj("partial", stop="max_tokens"))[0] == "MODEL"
    assert c(False, _cj("   "))[0] == "MODEL"
    item = dict(id="x", cat="math", sub="exact", prompt="2+2?", answer=4, kind="num", tol=0)
    rows = {cls: run.run_unit(item, "A", 0, "tune", "job", max_tries=2, sleep=NOSLEEP, callers=dict(claude=_const(cls), ollama=_const(cls)))
            for cls in ("OK", "MODEL", "INFRA")}
    assert rows["OK"]["status"] == "ok" and rows["OK"]["correct"] is True
    assert rows["MODEL"]["status"] == "model_fail" and rows["MODEL"]["correct"] is False        # a model failure scores as wrong
    assert rows["INFRA"]["status"] == "infra_fail" and rows["INFRA"]["correct"] is None and rows["INFRA"]["attempts"] == 2   # never scored
    code = dict(id="c", cat="code", sub="func", prompt="p", kind="code", answer=None, fn="f", tests=[[1]], expected=["2"])
    monkeypatch.setattr(run.sandbox, "run_code", lambda *a, **k: dict(status="infra", why="daemon down"))
    r = run.run_unit(code, "A", 0, "tune", "docker", max_tries=2, sleep=NOSLEEP, callers=dict(claude=_const("OK", "```python\ndef f(x): return 2*x\n```")))
    assert r["status"] == "infra_fail" and r["correct"] is None                 # a sandbox outage is infra, not a wrong answer


def test_retry():
    sleeps = []
    seq = iter(["INFRA", "INFRA", "OK"])
    r, n, errs = run.call_with_retry(lambda: dict(cls=next(seq), err="boom"), max_tries=5, sleep=sleeps.append, rng=random.Random(0))
    assert (r["cls"], n, len(sleeps), len(errs)) == ("OK", 3, 2, 2)
    assert 0 <= sleeps[0] <= 2.0 and 0 <= sleeps[1] <= 4.0                     # full jitter under an exponential cap
    sleeps.clear()
    r, n, _ = run.call_with_retry(lambda: dict(cls="INFRA", err="x"), max_tries=5, sleep=sleeps.append)
    assert (r["cls"], n, len(sleeps)) == ("INFRA", 5, 4)
    r, n, _ = run.call_with_retry(lambda: dict(cls="MODEL"), max_tries=5, sleep=sleeps.append)
    assert (r["cls"], n) == ("MODEL", 1)                                       # model failures are not retried

    class Top:
        @staticmethod
        def uniform(a, b): return b
    assert [run.backoff(k, rng=Top) for k in range(1, 8)] == [2, 4, 8, 16, 32, 60, 60]


def test_strict_parse():
    s = run.extract_strict
    assert s("work...\nANSWER: 42") == "42"
    assert s("**ANSWER:** 42\n\n") == "42"
    assert s("ANSWER: `abc`.") == "abc"
    assert s("ANSWER: 41\nactually 42") == ""                                  # not on the last line
    assert s("the answer is 42") == ""
    num = dict(kind="num", answer=42, tol=0)
    r = run.score(num, "I think it is 42")
    assert r["correct"] is False and r["fallback_correct"] is True and r["parsed"] is False   # fallback is diagnostic only
    r = run.score(num, "ANSWER: 42")
    assert r["correct"] is True and r["parsed"] is True
    assert run.code_of("```python\ndef f(): pass\n```") == "def f(): pass\n"
    assert run.code_of("def f(): pass") is None and run.code_of("def f(): pass", strict=False) == "def f(): pass"


def _mini_battery(tmp_path, split="tune", n=3):
    items = [dict(id=f"m{i}", split=split, cat="math", sub="exact", prompt=f"{i}+1?", answer=i + 1, kind="num", tol=0) for i in range(n)]
    p = tmp_path / f"battery_{split}.json"
    p.write_text(json.dumps(dict(header=dict(split=split, items_sha=run.json_sha(items)), items=items)), encoding="utf8")
    return p, items


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    bp, items = _mini_battery(tmp_path)
    monkeypatch.setattr(run, "battery_path", lambda s: bp)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "stale_check", lambda *a: None)
    monkeypatch.setattr(run, "claude_cli_version", lambda: "test")
    monkeypatch.setattr(run, "ollama_digest", lambda m=None: "sha256:test")
    monkeypatch.setattr(run, "frontend_prompt", lambda p: (p, {"retrieved": False}))
    for k in run.ENV_KEYS: monkeypatch.delenv(k, raising=False)
    return tmp_path, items


def _fake(fail_ids=(), calls=None):
    def claude(prompt):
        if calls is not None: calls.append(prompt)
        if any(prompt.startswith(f"{i}+") for i in fail_ids): return dict(cls="INFRA", text="", wall=0.0, err="503", tokens=None)
        return dict(cls="OK", text=f"ANSWER: {int(prompt.split('+')[0]) + 1}", wall=0.01, tokens=dict(total=7), cost_usd=0.0)
    return dict(claude=claude, ollama=claude)


def _rows(tmp_path, tag):
    return [json.loads(l) for l in open(tmp_path / "results" / tag / "raw.jsonl", encoding="utf8")]


def test_resume_no_dupes(offline_run):
    tmp_path, items = offline_run
    args = ["--split", "tune", "--cond", "A,B", "--tag", "t", "--k", "2", "--sandbox", "job", "--max-tries", "2", "--no-analyze"]
    assert run.main(args, callers=_fake(fail_ids=[1]), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    assert sum(r["status"] == "infra_fail" for r in _rows(tmp_path, "t")) == 4          # item 1: 2 conds x 2 reps, written but unscored
    calls = []
    assert run.main(args, callers=_fake(calls=calls), sleep=NOSLEEP) == run.EXIT_OK
    assert len(calls) == 4                                                              # only the infra_fail units were re-run
    final = [(r["id"], r["cond"], r["rep"]) for r in _rows(tmp_path, "t") if r["status"] in ("ok", "model_fail")]
    assert len(final) == len(set(final)) == 3 * 2 * 2                                   # one final row per (id, cond, rep)
    assert run.main(args, callers=_fake(calls=calls), sleep=NOSLEEP) == run.EXIT_OK and len(calls) == 4   # nothing left to do
    (tmp_path / "results" / "t" / ".lock").write_text("{}")                            # a second writer
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_LOCK
    with pytest.raises(SystemExit) as e:
        run.parse(["--split", "tune", "--tag", "t", "--cond", "A,E"])
    assert e.value.code == run.EXIT_USAGE


def test_manifest_mismatch(offline_run, monkeypatch):
    tmp_path, items = offline_run
    args = ["--split", "tune", "--cond", "A", "--tag", "m", "--k", "1", "--sandbox", "job", "--no-analyze"]
    assert run.main(args, callers=_fake(fail_ids=[0]), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    man = json.load(open(tmp_path / "results" / "m" / "manifest_tune.json"))
    assert {"git_sha", "git_dirty", "battery_sha", "corpus_sha", "frontend_sha", "prereg_sha", "models", "K", "sandbox", "env"} <= set(man)
    monkeypatch.setenv("BENCH_THINK", "9999")                                   # a different thinking budget is a different experiment
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_PREREG
    monkeypatch.delenv("BENCH_THINK")
    assert run.main(args[:7] + ["2"] + args[8:], callers=_fake(), sleep=NOSLEEP) == run.EXIT_PREREG     # K changed
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_OK
