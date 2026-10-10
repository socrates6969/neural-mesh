"""bench/tests/test_sandbox.py - the sandbox trust boundary (B1): expected outputs never enter the sandbox, the host scores.
Every attack below is a candidate whose honest answer is WRONG and which tries to make the host record a pass anyway. Each runs
in job mode (a real local child) and in docker mode with the docker CLI stubbed (sandbox._exec), so the docker-path logic is
exercised without a daemon. Run: python -m pytest bench/tests -q"""
import json, pathlib, subprocess, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import sandbox  # noqa: E402

SENTINEL = "EXPECTED-SENTINEL-6f1d"           # the expected output; it must never be visible inside the sandbox
TESTS, EXPECTED = [[3], [5]], [sandbox.canon(SENTINEL), sandbox.canon(SENTINEL)]

ATTACKS = {
    # reach the driver through __main__ and look for the expected values (they are not there any more)
    "import_main": "import __main__\n"
                   "def f(x):\n    return (__main__._P.get('expected') or ['nope'])[0]",
    "gc_scan": "import gc\n"
               "def f(x):\n    hits = [o for o in gc.get_objects() if isinstance(o, str) and o.startswith('EXPECTED-')]\n    return hits[0] if hits else 'nope'",
    # call the driver's own emitter with an old-style pass result
    "main_emit": "import __main__\n__main__._emit({'ok': True, 'bad': 0, 'n': 2})\ndef f(x): return 'nope'",
    # print fake pass lines: a forged nonce, and the real nonce in the old format
    "fake_pass_line": "import __main__, sys\nprint('\\n@@RESULT-0000000000000000@@{\"ok\": true, \"bad\": 0, \"n\": 2}')\n"
                      "print(__main__._NONCE + '{\"ok\": true, \"bad\": 0, \"n\": 2}')\ndef f(x): return 'nope'",
    # forge a well-formed result line with the real nonce, then leave before the driver writes its own
    "forge_then_exit": "import __main__, os, sys\n"
                       "sys.stdout.write('\\n' + __main__._NONCE + '{\"out\": [[\"v\", \"\\\\\"guess\\\\\"\"], [\"v\", \"\\\\\"guess\\\\\"\"]]}\\n')\n"
                       "sys.stdout.flush()\nos._exit(0)\ndef f(x): return 'nope'",
    "exit_before_output": "import os\nos._exit(0)\ndef f(x): return 'nope'",
    "exit_in_function": "import os\ndef f(x):\n    os._exit(0)",
    "sys_exit": "import sys\ndef f(x):\n    sys.exit(0)",
    # monkeypatch json: the old harness compared json.dumps(result) == expected inside the sandbox
    "patch_json": "import json, json.encoder\n"
                  "json.dumps = lambda *a, **k: '\"EXPECTED-SENTINEL-6f1d\"'\n"
                  "json.loads = lambda s, *a, **k: [0]\n"
                  "json.JSONEncoder.encode = lambda self, o: '{\"ok\": true}'\n"
                  "def f(x): return 'nope'",
    "patch_builtins": "import builtins\n"
                      "builtins.len = lambda x: 0\nbuiltins.isinstance = lambda *a: True\nbuiltins.callable = lambda x: True\n"
                      "builtins.type = lambda *a: type\n"
                      "def f(x): return 'nope'",
    # hijack the emitter and write the result twice (once rewritten)
    "double_write": "import __main__\n_orig = __main__._emit\n"
                    "def _twice(d):\n    __main__._write(1, ('\\n' + __main__._NONCE + __main__._dumps({'out': [['v', '1'], ['v', '1']]}) + '\\n').encode())\n    _orig(d)\n"
                    "__main__._emit = _twice\ndef f(x): return 'nope'",
    "stdout_twice": "import __main__, os\n"
                    "def f(x):\n    os.write(1, ('\\n' + __main__._NONCE + '{\"out\": [[\"v\", \"1\"], [\"v\", \"1\"]]}\\n').encode())\n    return 'nope'",
    # values json cannot encode count as wrong (type name), never as a crash or a pass
    "unserialisable": "def f(x):\n    return object()",
}


def fake_docker(calls=None, state=None, run_rc=None):
    """stand-in for the docker CLI: `docker run ...` runs the same program in a local `python -I -` child (no daemon needed)"""
    def _exec(cmd, input=None, timeout=None):
        if calls is not None: calls.append(cmd)
        if cmd[:2] == ["docker", "run"]:
            assert "python" in cmd and cmd[-2:] == ["-I", "-"]
            p = subprocess.run([sys.executable, "-I", "-"], input=input, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=timeout)
            return subprocess.CompletedProcess(cmd, p.returncode if run_rc is None else run_rc, p.stdout, p.stderr)
        if cmd[:2] == ["docker", "inspect"]:
            st = state if state is not None else dict(Status="exited", StartedAt="2026-10-10T00:00:00Z", Error="", OOMKilled=False)
            return subprocess.CompletedProcess(cmd, 0, json.dumps(st), "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return _exec


MODES = ["job", "docker-stub"]


def _run(mode, code, monkeypatch, tests=TESTS, expected=EXPECTED, **kw):
    if mode == "docker-stub":
        monkeypatch.setattr(sandbox, "_exec", fake_docker(**kw))
        return sandbox.run_code(code, "f", tests, expected, mode="docker")
    return sandbox.run_code(code, "f", tests, expected, mode="job")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", list(ATTACKS))
def test_attack_scores_wrong(name, mode, monkeypatch):
    r = _run(mode, ATTACKS[name], monkeypatch)
    assert r["status"] == "fail", (name, mode, r)          # a MODEL failure: never pass, never infra
    assert r["n"] == 2 and r["bad"] >= 1


@pytest.mark.parametrize("mode", MODES)
def test_honest_candidates(mode, monkeypatch):
    double = lambda xs: [sandbox.canon(2 * x[0]) for x in xs]      # noqa: E731
    assert _run(mode, "def f(x): return x * 2", monkeypatch, expected=double(TESTS))["status"] == "pass"
    r = _run(mode, "def f(x): return 6", monkeypatch, expected=double(TESTS))
    assert (r["status"], r["bad"]) == ("fail", 1)                  # f(3) right, f(5) wrong
    noisy = "import sys\ndef f(x):\n    print('ANSWER: lots of noise', x); sys.stderr.write('err\\n')\n    return {'b': [x, 1.5], 'a': (x,)}"
    exp = [sandbox.canon({"a": [x[0]], "b": [x[0], 1.5]}) for x in TESTS]
    assert _run(mode, noisy, monkeypatch, expected=exp)["status"] == "pass"    # host re-canonicalises (tuple -> list, key order)
    assert _run(mode, "def g(x): return x", monkeypatch)["why"].startswith("candidate: missing function")
    assert _run(mode, "def f(:", monkeypatch)["why"].startswith("candidate: compile:SyntaxError")
    assert _run(mode, "def f(x): raise ValueError('x')", monkeypatch)["status"] == "fail"


@pytest.mark.parametrize("mode", MODES)
def test_exit_codes_are_model_failures_in_job_mode(mode, monkeypatch):
    if mode != "job": pytest.skip("docker exit codes 125/126/127 are covered by the container-state tests")
    for rc in (1, 125, 126, 127):
        r = _run(mode, f"import os\nos._exit({rc})\ndef f(x): return 'nope'", monkeypatch)
        assert r["status"] == "fail", (rc, r)


def test_expected_never_enters_the_sandbox(monkeypatch):
    seen = []
    monkeypatch.setattr(sandbox, "_exec", fake_docker(calls=seen))
    sandbox.run_code("def f(x): return 1", "f", TESTS, EXPECTED, mode="docker")
    prog, nonce = sandbox.program("def f(x): return 1", "f", TESTS)
    assert SENTINEL not in prog and "expected" not in prog.split("_P = ")[1].split("\n")[0]
    assert all(SENTINEL not in " ".join(c) for c in seen)
    import inspect
    assert "expected" not in inspect.signature(sandbox.program).parameters
    assert sandbox.program("x", "f", [])[1] != nonce                   # a fresh nonce per run


def test_parse_outputs():
    n = "@@RESULT-abc@@"
    ok = n + json.dumps({"out": [["v", "1"], ["t", "set"]]})
    assert sandbox.parse_outputs("noise\n" + ok + "\nmore", n, 2)[0] == [["v", "1"], ["t", "set"]]
    assert sandbox.parse_outputs("", n, 2) == (None, "no result line")
    assert sandbox.parse_outputs(ok + "\n" + ok, n, 2) == (None, "2 result lines")
    assert sandbox.parse_outputs("x" + ok, n, 2)[0] is None                           # must start the line
    assert sandbox.parse_outputs(n + "{not json", n, 2)[1] == "malformed result line"
    assert sandbox.parse_outputs(n + json.dumps({"out": [], "ok": True}), n, 2)[1] == "malformed result object"
    assert sandbox.parse_outputs(n + json.dumps({"out": [["v", "1"]]}), n, 2)[1].startswith("malformed or wrong-length")
    assert sandbox.parse_outputs(n + json.dumps({"out": [["v", 1], ["v", "1"]]}), n, 2)[1] == "malformed output entry"
    assert sandbox.parse_outputs(n + json.dumps({"out": [["z", "1"], ["v", "1"]]}), n, 2)[1] == "malformed output entry"
    assert sandbox.parse_outputs(n + json.dumps({"err": "compile:X"}), n, 2)[1] == "candidate: compile:X"
    monkey = sandbox.MAX_OUT
    try:
        sandbox.MAX_OUT = 10
        assert sandbox.parse_outputs(ok, n, 2)[1] == "result line too long"
    finally:
        sandbox.MAX_OUT = monkey
    assert sandbox.score_outputs([["v", "1"], ["v", "2"], ["t", "set"], ["e", "ValueError"], ["v", "{bad"]], ["1", "3", '"set"', "1", "1"]) == 4
    assert sandbox.score_outputs([["v", '{"b": 1, "a": 2}']], [sandbox.canon({"a": 2, "b": 1})]) == 0


def test_run_code_args():
    with pytest.raises(ValueError):
        sandbox.run_code("", "f", [[1]], [], mode="job")
    with pytest.raises(ValueError):
        sandbox.run_code("", "f", [], [], mode="wasm")
