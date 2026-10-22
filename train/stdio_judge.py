"""Runs inside the grading container: /code/sol.py against /code/tests.json (stdin -> expected stdout).
Exit 0 only if every test passes. Output is compared token-by-token (whitespace-insensitive), floats with 1e-6 tolerance."""
import json, subprocess, sys

def same(out, exp):
    a, b = out.split(), exp.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if abs(fx - fy) > 1e-6 * max(1.0, abs(fy)):
            return False
    return True

spec = json.load(open("/code/tests.json"))
tl = spec["time_limit"]
for inp, exp in zip(spec["inputs"], spec["outputs"]):
    try:
        r = subprocess.run([sys.executable, "/code/sol.py"], input=inp, capture_output=True, text=True, timeout=tl)
    except subprocess.TimeoutExpired:
        sys.exit(3)
    if r.returncode != 0 or not same(r.stdout, exp):
        sys.exit(1)
sys.exit(0)