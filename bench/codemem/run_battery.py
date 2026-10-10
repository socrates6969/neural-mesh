"""bench battery with the codemem front-end as condition B (A stays the plain model). bench/frontend.py is NOT edited (its sha is the
pre-registration key); this wrapper replaces frontend_prompt before bench/run.py imports it. B = the REAL hook (codemem.exe hook) fed the
battery prompt with cwd = this repo; its stdout, if any, is prepended exactly as Claude Code would add it. Index = data/codemem/neural-mesh
(bench answer keys excluded by the repo config, see src/codemem/mod.rs default_config).
  python bench/codemem/run_battery.py --split tune --cond A,B --tag cm1 --k 1 [any other run.py flag]"""
import json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
NM = HERE.parents[1]
sys.path.insert(0, str(NM / "bench"))
import frontend  # noqa: E402
EXE = NM / "data" / "codemem" / "bin" / "codemem.exe"

def codemem_frontend(prompt):
    env = dict(os.environ, NEURAL_MESH_CODEMEM_LOG="off")
    r = subprocess.run([str(EXE), "hook"], input=json.dumps({"prompt": prompt, "cwd": str(NM), "session_id": "bench"}), capture_output=True, text=True, encoding="utf8", errors="replace", env=env, timeout=15)
    out = r.stdout.strip()
    if not out:
        return prompt, {"retrieved": False, "codemem": True}
    return out + "\n\n---\n" + prompt, {"retrieved": True, "codemem": True, "hook_chars": len(out)}

frontend.frontend_prompt = codemem_frontend
os.environ.setdefault("MESH_FRONTEND", "codemem")
import run  # noqa: E402  (binds the patched frontend_prompt)
sys.exit(run.main())
