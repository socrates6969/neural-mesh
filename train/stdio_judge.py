"""Runs inside the grading container. Only the test INPUTS are present (/code/inputs.json); expected outputs never
enter the container. Runs /code/sol.py once per input (with a minimal environment that excludes the result nonce)
and prints nonce + JSON list of {rc, out, timeout}; the host validates the structure and compares outputs."""
import json, os, subprocess, sys

NONCE = os.environ.pop("JUDGE_NONCE", "")
CHILD_ENV = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "PYTHONHASHSEED": "0"}

spec = json.load(open("/code/inputs.json"))
res = []
for inp in spec["inputs"]:
    try:
        r = subprocess.run([sys.executable, "/code/sol.py"], input=inp, capture_output=True, text=True,
                           timeout=spec["time_limit"], env=CHILD_ENV)
        res.append({"rc": r.returncode, "out": r.stdout[-200000:], "timeout": False})
    except subprocess.TimeoutExpired:
        res.append({"rc": None, "out": "", "timeout": True})
        break
sys.stdout.write("\n" + NONCE + json.dumps(res))