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


def lean_check(code: str, timeout_s: int = 120) -> str:
    code = str(code or "")
    if not code.strip():
        raise ValueError("code is required")
    needs = bool(re.search(r"^\s*import\s+Mathlib", code, re.M))
    image, err = _image(needs)
    if err:
        return err
    with_lib = needs and image != CORE  # plain lean is enough (and lighter) when Mathlib is not imported
    command = ["lake", "env", "lean", "/w/Main.lean"] if with_lib else ["lean", "/w/Main.lean"]
    out, err_txt, rc = run_sandbox(code, None, timeout_s=timeout_s, memory="4g" if with_lib else "1g", image=image,
                                   command=command, filename="Main.lean",
                                   workdir=MATHLIB_DIR if with_lib else "/w")
    text = (out + ("\n" + err_txt if err_txt.strip() else "")).strip()
    v, msgs = verdict(rc, text)
    lines = ["%s (%s, exit %d)" % (v, image, rc)]
    if v == "VERIFIED" and not msgs:
        lines.append("Lean accepted the file with no errors or warnings.")
    lines += msgs or ([text[-3000:]] if text else [])
    if v == "TIMEOUT":
        lines.append("killed after %s s" % timeout_s)
    return cap("\n".join(lines))
