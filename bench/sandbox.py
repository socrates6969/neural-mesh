"""bench/sandbox.py - run model-written code against hidden tests, contained, and compare to PRECOMPUTED expected outputs.

The reference implementation is never executed here: battery.py computes `expected` (canonical JSON of ref(*args)) at build
time, so the only untrusted code that runs is the candidate, and it runs in one of two modes:

  docker  (the only mode allowed for HELD/CONF): `docker run --rm -i` on a digest-pinned python:3.12-slim with
          --network none --read-only --tmpfs /tmp:size=16m --memory 256m --memory-swap 256m --cpus 1 --pids-limit 64
          --cap-drop ALL --security-opt no-new-privileges --user 65534, interpreter `python -I -`, program piped on stdin.
          A host-side timeout of 20 s ends the container with `docker kill`.
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
DOCKER_FLAGS = ["--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m", "--memory", f"{MEM_MB}m", "--memory-swap", f"{MEM_MB}m",
                "--cpus", "1", "--pids-limit", "64", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534",
                "--workdir", "/tmp", "--log-driver", "none"]
DOCKER_INFRA_RC = {125, 126, 127}      # docker run itself failed (daemon / image / exec), not the program

# The program sent on stdin. %(payload)s is a Python string literal of the JSON payload; %(nonce)s marks the result line so that
# ordinary prints of the candidate cannot be mistaken for it.
HARNESS = r'''
import json, os, sys
_P = json.loads(%(payload)s)
_WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_NONCE = %(nonce)r
def _emit(d):
    sys.__stdout__.write("\n" + _NONCE + json.dumps(d) + "\n"); sys.__stdout__.flush()
if _P.get("audit"):
    def _hook(ev, args, _deny=("socket.", "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
                                "os.startfile", "ctypes.", "winreg.", "os.remove", "os.rename", "os.rmdir", "os.mkdir", "shutil.")):
        if ev == "open":
            mode, flags = (list(args) + [None, None])[1:3]
            if (mode is not None and any(c in str(mode) for c in "wax+")) or (mode is None and isinstance(flags, int) and flags & _WFLAGS):
                raise PermissionError("sandbox: write refused")
        if ev.startswith(_deny):
            raise PermissionError("sandbox: " + ev + " refused")
    sys.addaudithook(_hook)
g = {"__name__": "candidate"}
try:
    exec(compile(_P["code"], "<candidate>", "exec"), g)
except BaseException as e:
    _emit({"ok": False, "why": "compile:" + repr(e)[:120]}); sys.exit(0)
if not callable(g.get(_P["fn"])):
    _emit({"ok": False, "why": "missing function " + _P["fn"]}); sys.exit(0)
bad = 0
for args, exp in zip(_P["tests"], _P["expected"]):
    try:
        got = json.dumps(g[_P["fn"]](*json.loads(json.dumps(args))), sort_keys=True, default=str)
    except BaseException as e:
        got = "EXC:" + repr(e)[:60]
    if got != exp: bad += 1
_emit({"ok": bad == 0, "bad": bad, "n": len(_P["tests"])})
'''


def canon(v):
    """canonical form used for `expected` (and for the candidate's result inside the harness)"""
    return json.dumps(v, sort_keys=True, default=str)


def program(code, fn, tests, expected, audit=False):
    nonce = "@@RESULT-" + secrets.token_hex(8) + "@@"
    payload = json.dumps(dict(code=code, fn=fn, tests=tests, expected=expected, audit=audit))
    return HARNESS % dict(payload=repr(payload), nonce=nonce), nonce


def parse_result(out, nonce):
    for line in reversed(out.splitlines()):
        if line.startswith(nonce):
            try: return json.loads(line[len(nonce):])
            except Exception: return None
    return None


# ------------------------------------------------------------------ docker
def docker_available(timeout=20):
    """(ok, why): the daemon answers and the pinned image is present locally (pull it with ensure_image())."""
    if not shutil.which("docker"): return False, "docker CLI not found"
    try:
        p = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0: return False, "daemon: " + (p.stderr or p.stdout).strip()[:200]
        p = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0: return False, f"image {IMAGE} not present (run: docker pull {IMAGE})"
    except Exception as e:
        return False, repr(e)[:200]
    return True, "ok"


def ensure_image(timeout=600):
    p = subprocess.run(["docker", "pull", IMAGE], capture_output=True, text=True, timeout=timeout)
    return p.returncode == 0, (p.stdout + p.stderr)[-300:]


def _run_docker(prog, nonce, timeout):
    name = "bench-sbx-" + uuid.uuid4().hex[:12]
    cmd = ["docker", "run", "--rm", "-i", "--name", name, *DOCKER_FLAGS, IMAGE, "python", "-I", "-"]
    t0 = time.time()
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf8", errors="replace")
    except Exception as e:
        return dict(status="infra", why="spawn:" + repr(e)[:150], secs=0.0)
    try:
        out, err = p.communicate(prog, timeout=timeout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)
        try: p.communicate(timeout=15)
        except Exception: p.kill()
        return dict(status="timeout", why=f"host timeout {timeout}s", secs=time.time() - t0)
    secs = time.time() - t0
    res = parse_result(out, nonce)
    if res is not None:
        return dict(status="pass" if res.get("ok") else "fail", bad=res.get("bad"), n=res.get("n"), why=res.get("why"), secs=secs)
    if p.returncode in DOCKER_INFRA_RC or "Error response from daemon" in err or "error during connect" in err:
        return dict(status="infra", why=f"docker rc={p.returncode}: {err.strip()[:200]}", secs=secs)
    return dict(status="fail", why=f"no result (rc={p.returncode}): {err.strip()[-160:]}", secs=secs)   # e.g. OOM-killed (137)


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


def _run_job(prog, nonce, timeout):
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
        res = parse_result(out, nonce)
        secs = time.time() - t0
        if res is None: return dict(status="fail", why=f"no result (rc={p.returncode}): {err.strip()[-160:]}", secs=secs)
        return dict(status="pass" if res.get("ok") else "fail", bad=res.get("bad"), n=res.get("n"), why=res.get("why"), secs=secs)
    finally:
        if h is not None:
            import ctypes
            k32.CloseHandle(ctypes.c_void_p(h))     # KILL_ON_JOB_CLOSE: anything the candidate left running dies here
        shutil.rmtree(cwd, ignore_errors=True)


def run_code(code, fn, tests, expected, mode="docker", timeout=TIMEOUT_S):
    """Run candidate `code` defining `fn` on `tests` (list of arg lists) and compare with `expected` (list of canon() strings)."""
    if mode not in ("docker", "job"): raise ValueError(mode)
    if len(tests) != len(expected): raise ValueError("tests/expected length mismatch")
    prog, nonce = program(code, fn, tests, expected, audit=(mode == "job"))
    return _run_docker(prog, nonce, timeout) if mode == "docker" else _run_job(prog, nonce, timeout)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true", help="report whether docker mode is usable")
    ap.add_argument("--pull", action="store_true", help="docker pull the pinned image")
    a = ap.parse_args()
    if a.pull: print(ensure_image())
    ok, why = docker_available(); print(json.dumps(dict(docker=ok, why=why, image=IMAGE))); sys.exit(0 if ok else 4)
