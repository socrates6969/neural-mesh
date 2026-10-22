"""Runs inside the grading container. Only the test INPUTS are present (/code/inputs.json); expected outputs never
enter the container. Runs /code/sol.py once per input and prints a JSON list of {rc, out, timeout} to stdout,
which the host compares against the expected outputs."""
import json, subprocess, sys

spec = json.load(open("/code/inputs.json"))
res = []
for inp in spec["inputs"]:
    try:
        r = subprocess.run([sys.executable, "/code/sol.py"], input=inp, capture_output=True, text=True, timeout=spec["time_limit"])
        res.append({"rc": r.returncode, "out": r.stdout[-200000:], "timeout": False})
    except subprocess.TimeoutExpired:
        res.append({"rc": None, "out": "", "timeout": True})
        break
sys.stdout.write("\n" + spec["nonce"] + json.dumps(res))