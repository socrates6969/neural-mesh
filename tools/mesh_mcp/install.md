# Installing mesh-tools

## 1. Image

    cd C:\Users\mariu\ruv\hive\runs\agent-catalog\grader
    docker build -t meshtrain-grader:2 .

`meshtrain-grader:1` is a separate tag and is not changed. Rebuild it with
`docker build -f Dockerfile.v1 -t meshtrain-grader:1 .` if it is ever removed.

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
