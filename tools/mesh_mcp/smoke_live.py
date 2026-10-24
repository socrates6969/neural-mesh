"""Live smoke test: hits each literature API once and exercises every tool through the real server.

    python -m tools.mesh_mcp.smoke_live            # run against the live APIs and Docker
    python -m tools.mesh_mcp.smoke_live --record   # also refresh tests/fixtures from the live API responses

Needs network (for the APIs) and Docker with the grader image. Not part of the offline unit tests.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from tools.mesh_mcp import literature

FIX = Path(__file__).resolve().parent / "tests" / "fixtures"
QUERY = "transformer attention"
NAMES = [("api.semanticscholar.org", "s2_search.json"), ("export.arxiv.org", "arxiv_query.xml"),
         ("esearch.fcgi", "pubmed_esearch.json"), ("esummary.fcgi", "pubmed_esummary.json"),
         ("efetch.fcgi", "pubmed_efetch.xml")]


class Recorder:
    def __init__(self):
        self.inner = literature.Fetcher()

    def __call__(self, url, headers=None):
        body = self.inner(url, headers)
        for key, name in NAMES:
            if key in url:
                (FIX / name).write_bytes(body)
        time.sleep(0.4)  # stay polite to NCBI (3 req/s without a key)
        return body


def rpc(proc, mid, method, params=None):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def main() -> int:
    record = "--record" in sys.argv
    ok = True
    fetch = Recorder() if record else None
    for src in literature.SOURCES:
        t = time.time()
        out = literature.literature_search(QUERY, [src], 2, fetch=fetch)
        good = ("[%s]" % src) in out and "unavailable" not in out
        ok &= good
        print("%-17s %s %.1fs  %s" % (src, "OK  " if good else "FAIL", time.time() - t, out.splitlines()[1][:90]))
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen([sys.executable, "-m", "tools.mesh_mcp.server"], cwd=root, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        rpc(proc, 1, "initialize")
        names = [t["name"] for t in rpc(proc, 2, "tools/list")["result"]["tools"]]
        print("tools:", ", ".join(names))
        r = rpc(proc, 3, "tools/call", {"name": "math_solve", "arguments": {
            "code": "import sympy as sp, z3\nx = sp.symbols('x')\nprint(sp.integrate(x**2, (x, 0, 3)))\n"
                    "a = z3.Int('a'); s = z3.Solver(); s.add(a * a == 49, a > 0); print(s.check(), s.model()[a])"}})
        text = r["result"]["content"][0]["text"]
        good = "9" in text and "sat" in text and "7" in text
        ok &= good
        print("math_solve        %s %s" % ("OK  " if good else "FAIL", " ".join(text.split())[:90]))
        r = rpc(proc, 4, "tools/call", {"name": "stats_test", "arguments": {
            "kind": "welch", "data_json": {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}}})
        text = r["result"]["content"][0]["text"]
        good = "EFFECT SIZE" in text
        ok &= good
        print("stats_test        %s %s" % ("OK  " if good else "FAIL", text.splitlines()[0]))
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
