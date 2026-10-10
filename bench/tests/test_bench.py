"""bench/tests/test_bench.py - offline tests for the bench battery (no model calls, no network except a local docker daemon).
Run: python -m pytest bench/tests -q"""
import json, os, pathlib, sys, time
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
