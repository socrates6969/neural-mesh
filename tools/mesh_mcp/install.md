# Installing mesh-tools

## 1. Image

    cd C:\Users\mariu\ruv\hive\runs\agent-catalog\grader
    docker build -t meshtrain-grader:2 .

`meshtrain-grader:1` is a separate tag and is not changed. Rebuild it with
`docker build -f Dockerfile.v1 -t meshtrain-grader:1 .` if it is ever removed.

Lean 4 for `lean_check` (Lean v4.35.0 via elan v4.2.4, toolchain pinned to match Mathlib v4.35.0):

    cd tools\mesh_mcp\lean
    docker build -f Dockerfile.core -t meshtrain-lean:core .         # core Lean only, about 1 GB
    docker build -f Dockerfile.mathlib -t meshtrain-lean:mathlib .   # optional: prebuilt Mathlib from its cache, several GB

Built 2026-10-24: core 4.35 GB (927 MB compressed); mathlib 15.2 GB (3.5 GB compressed), cache download worked.
Measured: a bare `import Mathlib` (all 7.1 GB of oleans) takes 403-482 s at 6 GB / 2 CPUs (690 s at 4 GB / 1 CPU) and
does not speed up on reuse; a targeted import (`import Mathlib.Analysis.SpecialFunctions.Pow.Real`) takes ~24 s;
core-only checks take a few seconds. lean_check budgets: core 120 s (max 300), Mathlib modules 240 s / 4 GB
(max 600), full Mathlib 600 s / 6 GB / 2 CPUs (max 900). The 6 GB run needs most of the 8 GB Docker VM.
Lean is not installed on the Windows host; it only runs inside these Linux images.

## 2. Register for the user (all projects)

    powershell -File tools\mesh_mcp\install.ps1 -DryRun     # prints the command only
    powershell -File tools\mesh_mcp\install.ps1             # runs: claude mcp add-json --scope user mesh-tools {...}

`-RepoRoot` defaults to the checkout the script is in (the worktree until `exp/mesh-tools` is merged); after the
merge use `-RepoRoot C:\Users\mariu\neural-mesh`. The script sets `PYTHONPATH` to the repo root in the server's
`env`, so no working directory is needed. Equivalent manual command:

    claude mcp add-json --scope user mesh-tools '{"type":"stdio","command":"C:\\Users\\mariu\\AppData\\Local\\Programs\\Python\\Python312\\python.exe","args":["-m","tools.mesh_mcp.server"],"env":{"PYTHONPATH":"C:\\Users\\mariu\\neural-mesh","MESH_TOOLS_IMAGE":"meshtrain-grader:2","PYTHONIOENCODING":"utf-8"}}'

Check with `claude mcp get mesh-tools`; remove with `claude mcp remove --scope user mesh-tools`.
Tools then appear to agents as `mcp__mesh-tools__<tool>`. Subagents whose definition lists explicit `tools:`
need those names added (or no `tools:` line) to see them.

## 3. Training harness (lead's decision; not changed here)

`train/runner.py` runs agents with `--strict-mcp-config` and an `mcp.json` holding only the `sandbox` server, and
`clean_env()` strips the environment, so `PYTHONPATH` must be passed in the server's `env`. To offer mesh tools in
evals, add a second server and allow its tools:

```python
mcp = {"mcpServers": {
    "sandbox": {"command": sys.executable, "args": [SANDBOX_SERVER, str(tmp)]},
    "mesh": {"command": sys.executable, "args": ["-m", "tools.mesh_mcp.server"],
             "env": {"PYTHONPATH": REPO_ROOT, "MESH_TOOLS_ENABLE": "math_solve,stats_test",
                     "MESH_TOOLS_IMAGE": "meshtrain-grader:2"}}}}
ALLOWED += ["mcp__mesh__math_solve", "mcp__mesh__stats_test"]   # and add them to the agent spec "tools"
```

Recommendation: in graded evals expose only offline tools (`math_solve`, `stats_test`, optionally
`causal_analysis`). `literature_search` reaches the internet and can surface benchmark answers, and
`workspace_post`/`report_imrad` write shared state, which would leak between eval tasks.
The harness's own `run_python` keeps using `MESHTRAIN_GRADER_IMAGE` (default `:1`); set
`MESHTRAIN_GRADER_IMAGE=meshtrain-grader:2` to give it the same libraries.
