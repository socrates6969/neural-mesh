"""bench/sandbox.py - run model-written code on the hidden test inputs, contained, and score its outputs HERE, in the host process.

Trust boundary. The only untrusted code is the candidate. Inside the sandbox nothing but the candidate runs, on the test
INPUTS; the precomputed `expected` outputs (battery.py: canonical JSON of ref(*args)) never enter the sandbox. The sandbox
program prints the candidate's raw outputs once, as one JSON line behind a random per-run nonce; the host takes exactly that
line, re-canonicalises every value and compares it with `expected`. Whatever the candidate does to the interpreter it shares
with the small driver (import __main__, monkeypatch json/builtins, print fake result lines, os._exit, write twice) it can only
make its own output missing, duplicated or malformed, and the host scores all of those as wrong (a MODEL failure, never
INFRA). To pass it has to return the right values, which is the task.

Output protocol (one line on stdout): <nonce>{"out": [[tag, val], ...]} with one entry per test: tag "v" (val = the value as
JSON text), "t" (val = type name of a result json cannot encode; always wrong) or "e" (val = exception type name; always
wrong); or <nonce>{"err": "..."} when the code does not compile or does not define the function.

Modes:
  docker  (the only mode allowed for HELD/CONF): `docker run -i` on a digest-pinned python:3.12-slim with --network none
          --read-only --tmpfs /tmp:size=16m --memory 256m --memory-swap 256m --cpus 1 --pids-limit 64 --cap-drop ALL
          --security-opt no-new-privileges --user 65534, interpreter `python -I -`, program piped on stdin. A host-side timeout
          of 20 s ends the container with `docker kill`.
  job     (TUNE only; NOT a security boundary): a local `python -I -` child with an empty environment (plus SYSTEMROOT, which
          the Windows interpreter needs to start), a fresh temp cwd, a Windows Job Object (256 MB per process,
          KILL_ON_JOB_CLOSE, at most 8 active processes) and an audit hook that refuses file writes, sockets and process
          creation. A determined program can get past an audit hook; this mode only stops accidents.

run_code(code, fn, tests, expected, mode) -> dict(status=pass|fail|timeout|infra, bad, n, why, secs)
  pass/fail/timeout are MODEL outcomes (timeout is scored as wrong); infra means the sandbox itself did not work (docker
  daemon down, image missing, ...) and the caller must retry instead of scoring."""
import json, os, secrets, shutil, subprocess, sys, tempfile, time, uuid

IMAGE = "python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1"   # index digest, 2026-10-10
TIMEOUT_S = 20
MEM_MB = 256
MAX_OUT = 8 << 20                       # a result line longer than this is malformed (scored wrong)
DOCKER_FLAGS = ["--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m", "--memory", f"{MEM_MB}m", "--memory-swap", f"{MEM_MB}m",
                "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534",
                "--workdir", "/tmp", "--log-driver", "none"]
DOCKER_INFRA_RC = {125, 126, 127}      # docker run itself failed (daemon / image / exec), not the program

# The program sent on stdin: the candidate and the test inputs, nothing else. %(payload)s is a Python string literal of the JSON
# payload {code, fn, tests, audit}; %(nonce)s marks the one result line. The driver keeps its own references to json.dumps and
# os.write, but nothing here is a secret: the candidate may find and break all of it, and can only lose by doing so.
HARNESS = r'''
import json as _json, os as _os, sys as _sys
_P = _json.loads(%(payload)s)
_NONCE = %(nonce)r
_dumps, _write, _exit = _json.dumps, _os.write, _os._exit
def _emit(d):
    try: _sys.stdout.flush()
    except BaseException: pass
    b = ("\n" + _NONCE + _dumps(d) + "\n").encode("utf8")
    while b: b = b[_write(1, b):]
    _exit(0)
if _P.get("audit"):
    _WFLAGS = _os.O_WRONLY | _os.O_RDWR | _os.O_CREAT | _os.O_APPEND | _os.O_TRUNC
    def _hook(ev, args, _deny=("socket.", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                                "os.startfile", "ctypes.", "winreg.", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "shutil.")):
        if ev == "open":
            mode, flags = (list(args) + [None, None])[1:3]
            if (mode is not None and any(c in str(mode) for c in "wax+")) or (mode is None and isinstance(flags, int) and flags & _WFLAGS):
                raise PermissionError("sandbox: write refused")
        if ev.startswith(_deny):
            raise PermissionError("sandbox: " + ev + " refused")
    _sys.addaudithook(_hook)
_g = {"__name__": "candidate"}
try:
    exec(compile(_P["code"], "<candidate>", "exec"), _g)
except BaseException as _e:
    _emit({"err": "compile:" + type(_e).__name__})
_f = _g.get(_P["fn"])
if not callable(_f):
    _emit({"err": "missing function " + _P["fn"]})
_out = []
for _a in _P["tests"]:
    try:
        _v = _f(*_json.loads(_dumps(_a)))
    except BaseException as _e:
        _out.append(["e", type(_e).__name__]); continue
    try:
        _out.append(["v", _dumps(_v, sort_keys=True)])
    except BaseException:
        _out.append(["t", type(_v).__name__])
_emit({"out": _out})
'''


def canon(v):
    """canonical form used for `expected`; the host re-canonicalises every candidate value with it before comparing"""
    return json.dumps(v, sort_keys=True, default=str)


def program(code, fn, tests, audit=False):
    """-> (program text, nonce). The expected outputs are deliberately not a parameter: they never leave the host."""
    nonce = "@@RESULT-" + secrets.token_hex(16) + "@@"
    payload = json.dumps(dict(code=code, fn=fn, tests=tests, audit=audit))
    return HARNESS % dict(payload=repr(payload), nonce=nonce), nonce


def parse_outputs(out, nonce, n):
    """-> (outputs, why). outputs is a list of n [tag, val] pairs, or None when the result is missing, duplicated or malformed
    (all scored as wrong). Only a line that STARTS with this run's nonce counts."""
    hits = [l for l in (out or "").splitlines() if l.startswith(nonce)]
    if not hits: return None, "no result line"
    if len(hits) > 1: return None, f"{len(hits)} result lines"
    body = hits[0][len(nonce):]
    if len(body) > MAX_OUT: return None, "result line too long"
    try: d = json.loads(body)
    except Exception: return None, "malformed result line"
    if not isinstance(d, dict) or len(d) != 1: return None, "malformed result object"
    if "err" in d: return None, "candidate: " + str(d["err"])[:150]
    o = d.get("out")
    if not isinstance(o, list) or len(o) != n: return None, "malformed or wrong-length output list"
    for e in o:
        if not (isinstance(e, list) and len(e) == 2 and e[0] in ("v", "t", "e") and isinstance(e[1], str)): return None, "malformed output entry"
    return o, None


def score_outputs(outputs, expected):
    """number of tests whose output is not exactly the expected canonical value (type names and exceptions are always wrong)"""
    bad = 0
    for (tag, val), exp in zip(outputs, expected):
        if tag != "v": bad += 1; continue
        try: got = canon(json.loads(val))
        except Exception: bad += 1; continue
        bad += got != exp
    return bad


def verdict(out, nonce, expected, secs, rc=None, err=""):
    """host-side scoring of one finished sandbox run (a MODEL outcome: pass or fail, never infra)"""
    outputs, why = parse_outputs(out, nonce, len(expected))
    if outputs is None:
        return dict(status="fail", bad=len(expected), n=len(expected), why=f"{why} (rc={rc}): {(err or '').strip()[-160:]}", secs=secs)
    bad = score_outputs(outputs, expected)
    return dict(status="pass" if bad == 0 else "fail", bad=bad, n=len(expected), why=None, secs=secs)


# ------------------------------------------------------------------ docker
def _exec(cmd, input=None, timeout=None):
    """one docker CLI call -> CompletedProcess (raises subprocess.TimeoutExpired / OSError). The tests stub this."""
    return subprocess.run(cmd, input=input, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=timeout)


def docker_available(timeout=20):
    """(ok, why): the daemon answers and the pinned image is present locally (pull it with ensure_image())."""
    if not shutil.which("docker"): return False, "docker CLI not found"
    try:
        p = _exec(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=timeout)
        if p.returncode != 0: return False, "daemon: " + (p.stderr or p.stdout).strip()[:200]
        p = _exec(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], timeout=timeout)
        if p.returncode != 0: return False, f"image {IMAGE} not present (run: docker pull {IMAGE})"
    except Exception as e:
        return False, repr(e)[:200]
    return True, "ok"


def ensure_image(timeout=600):
    p = _exec(["docker", "pull", IMAGE], timeout=timeout)
    return p.returncode == 0, (p.stdout + p.stderr)[-300:]


def _run_docker(prog, nonce, expected, timeout):
    name = "bench-sbx-" + uuid.uuid4().hex[:12]
    cmd = ["docker", "run", "--rm", "-i", "--name", name, *DOCKER_FLAGS, IMAGE, "python", "-I", "-"]
    t0 = time.time()
    try:
        p = _exec(cmd, input=prog, timeout=timeout)
    except subprocess.TimeoutExpired:
        try: _exec(["docker", "kill", name], timeout=30)
        except Exception: pass
        return dict(status="timeout", why=f"host timeout {timeout}s", secs=time.time() - t0)
    except Exception as e:
        return dict(status="infra", why="spawn:" + repr(e)[:150], secs=0.0)
    secs = time.time() - t0
    if parse_outputs(p.stdout, nonce, len(expected))[0] is None and (
            p.returncode in DOCKER_INFRA_RC or "Error response from daemon" in p.stderr or "error during connect" in p.stderr):
        return dict(status="infra", why=f"docker rc={p.returncode}: {p.stderr.strip()[:200]}", secs=secs)
    return verdict(p.stdout, nonce, expected, secs, p.returncode, p.stderr)      # e.g. OOM-killed (137): no result -> fail


# ------------------------------------------------------------------ job object (Windows) / rlimit (POSIX) fallback
def _job_object(mem_mb, max_procs):
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                      "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO_COUNTERS), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    h = k32.CreateJobObjectW(None, None)
    if not h: raise OSError(ctypes.get_last_error(), "CreateJobObjectW")
    info = EXT()
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOB_OBJECT_LIMIT_JOB_MEMORY, KILL_ON_CLOSE = 0x8, 0x100, 0x200, 0x2000
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_JOB_MEMORY | KILL_ON_CLOSE
    info.BasicLimitInformation.ActiveProcessLimit = max_procs
    info.ProcessMemoryLimit = mem_mb << 20; info.JobMemoryLimit = (mem_mb << 20) * 2
    if not k32.SetInformationJobObject(wintypes.HANDLE(h), 9, ctypes.byref(info), ctypes.sizeof(info)):   # 9 = JobObjectExtendedLimitInformation
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject")
    return k32, h


def _run_job(prog, nonce, expected, timeout):
    t0 = time.time()
    cwd = tempfile.mkdtemp(prefix="bench-job-")
    env = {"SYSTEMROOT": os.environ["SYSTEMROOT"]} if sys.platform == "win32" and "SYSTEMROOT" in os.environ else {}
    k32 = h = None
    try:
        kw = {}
        if sys.platform != "win32":
            def _limits():
                import resource
                resource.setrlimit(resource.RLIMIT_AS, (MEM_MB << 20, MEM_MB << 20)); resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            kw["preexec_fn"] = _limits
        try:
            p = subprocess.Popen([sys.executable, "-I", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, env=env,
                                 text=True, encoding="utf8", errors="replace", **kw)
        except Exception as e:
            return dict(status="infra", why="spawn:" + repr(e)[:150], secs=0.0)
        if sys.platform == "win32":
            try:
                import ctypes
                k32, h = _job_object(MEM_MB, 8)
                PROCESS_ALL_ACCESS = 0x1F0FFF
                ph = k32.OpenProcess(PROCESS_ALL_ACCESS, False, p.pid)
                ok = k32.AssignProcessToJobObject(ctypes.c_void_p(h), ctypes.c_void_p(ph)); k32.CloseHandle(ctypes.c_void_p(ph))
                if not ok: raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject")
            except Exception as e:     # the child is still blocked on stdin, so nothing untrusted has run yet
                p.kill(); p.communicate()
                return dict(status="infra", why="job object: " + repr(e)[:150], secs=time.time() - t0)
        try:
            out, err = p.communicate(prog, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            if h is not None:
                import ctypes
                k32.TerminateJobObject(ctypes.c_void_p(h), 1)
            try: p.communicate(timeout=10)
            except Exception: pass
            return dict(status="timeout", why=f"host timeout {timeout}s", secs=time.time() - t0)
        return verdict(out, nonce, expected, time.time() - t0, p.returncode, err)
    finally:
        if h is not None:
            import ctypes
            k32.CloseHandle(ctypes.c_void_p(h))     # KILL_ON_JOB_CLOSE: anything the candidate left running dies here
        shutil.rmtree(cwd, ignore_errors=True)


def run_code(code, fn, tests, expected, mode="docker", timeout=TIMEOUT_S):
    """Run candidate `code` defining `fn` on `tests` (list of arg lists); compare its outputs with `expected` (list of canon()
    strings) here on the host. `expected` is never sent into the sandbox."""
    if mode not in ("docker", "job"): raise ValueError(mode)
    if len(tests) != len(expected): raise ValueError("tests/expected length mismatch")
    prog, nonce = program(code, fn, tests, audit=(mode == "job"))
    return _run_docker(prog, nonce, expected, timeout) if mode == "docker" else _run_job(prog, nonce, expected, timeout)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true", help="report whether docker mode is usable")
    ap.add_argument("--pull", action="store_true", help="docker pull the pinned image")
    a = ap.parse_args()
    if a.pull: print(ensure_image())
    ok, why = docker_available(); print(json.dumps(dict(docker=ok, why=why, image=IMAGE))); sys.exit(0 if ok else 4)
