"""lean_check: type-check a Lean 4 snippet in a network-less, read-only container (neuro-symbolic proof checking).

Images: MESH_LEAN_IMAGE (default meshtrain-lean:mathlib when built, else meshtrain-lean:core). Snippets that
`import Mathlib` need the mathlib image; core Lean handles everything else. Definitions live in tools/mesh_mcp/lean/.
"""
from __future__ import annotations

import os
import re

from tools.mesh_mcp.common import cap, docker_available, run_sandbox

CORE, MATHLIB = "meshtrain-lean:core", "meshtrain-lean:mathlib"
MATHLIB_DIR = "/home/runner/mathlib4"
_MSG = re.compile(r"^/w/Main\.lean:(\d+):(\d+): (error|warning|info): ?(.*)$")
# Measured 2026-10-24 (meshtrain-lean:mathlib, Docker Desktop VM 8 GB): `import Mathlib` (all 7 GB of oleans) took
# 403-482 s at 6 GB / 2 CPUs and does not get faster on reuse; a targeted import such as
# `import Mathlib.Analysis.SpecialFunctions.Pow.Real` took 24 s at 4 GB / 1 CPU. Core Lean: a few seconds.
BUDGET = {"core": (120, 300, "1g", 1), "mathlib_part": (240, 600, "4g", 1), "mathlib_full": (600, 900, "6g", 2)}


def _image(needs_mathlib: bool) -> tuple[str | None, str | None]:
    forced = os.environ.get("MESH_LEAN_IMAGE")
    if forced:
        return forced, None
    if docker_available(MATHLIB):
        return MATHLIB, None
    if needs_mathlib:
        return None, ("Mathlib is not available (image %s not built); only core Lean is installed. "
                      "Remove `import Mathlib` and use core lemmas (Nat.*, List.*, omega, decide, simp)." % MATHLIB)
    if docker_available(CORE):
        return CORE, None
    return None, "Lean is not installed: build tools/mesh_mcp/lean/Dockerfile.core (see README)."


def verdict(rc: int, out: str) -> tuple[str, list[str]]:
    msgs = [m.group(0)[len("/w/"):] for m in map(_MSG.match, out.splitlines()) if m]
    errors = [m for m in msgs if ": error:" in m]
    if rc == 124:
        return "TIMEOUT", msgs
    if errors or rc != 0:
        return "FAILED", msgs
    if any(re.search(r"declaration uses .sorry.", m) for m in msgs):  # quoted 'sorry' or `sorry` by version
        return "INCOMPLETE (uses sorry)", msgs
    return "VERIFIED", msgs


def kind_of(code: str) -> str:
    if re.search(r"^\s*import\s+Mathlib\s*$", code, re.M):
        return "mathlib_full"
    if re.search(r"^\s*import\s+Mathlib\.", code, re.M):
        return "mathlib_part"
    return "core"


def lean_check(code: str, timeout_s: int | None = None) -> str:
    code = str(code or "")
    if not code.strip():
        raise ValueError("code is required")
    kind = kind_of(code)
    image, err = _image(kind != "core")
    if err:
        return err
    with_lib = kind != "core" and image != CORE  # plain lean is enough (and lighter) without Mathlib
    default_s, max_s, memory, cpus = BUDGET[kind if with_lib else "core"]
    timeout_s = int(timeout_s or default_s)
    command = ["lake", "env", "lean", "/w/Main.lean"] if with_lib else ["lean", "/w/Main.lean"]
    out, err_txt, rc = run_sandbox(code, None, timeout_s=timeout_s, memory=memory, image=image, command=command,
                                   filename="Main.lean", workdir=MATHLIB_DIR if with_lib else "/w", cpus=cpus,
                                   max_timeout_s=max_s)
    text = (out + ("\n" + err_txt if err_txt.strip() else "")).strip()
    v, msgs = verdict(rc, text)
    lines = ["%s (%s, exit %d)" % (v, image, rc)]
    if v == "VERIFIED" and not msgs:
        lines.append("Lean accepted the file with no errors or warnings.")
    lines += msgs or ([text[-3000:]] if text else [])
    if v == "TIMEOUT":
        lines.append("killed after %s s" % min(timeout_s, max_s))
        if kind == "mathlib_full":
            lines.append("`import Mathlib` loads all of Mathlib (about 7-8 min here). Import only the modules you "
                         "need, e.g. `import Mathlib.Analysis.SpecialFunctions.Pow.Real` (about 25 s).")
    return cap("\n".join(lines))
