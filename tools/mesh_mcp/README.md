# mesh-tools MCP server

A stdio MCP server (newline-delimited JSON-RPC, no MCP SDK) that gives every agent research, math, statistics,
causal-inference, shared-workspace and report tools. Python 3.12, standard library only on the host side; all
model-written code and all data analysis run in the network-less grader container `meshtrain-grader:2`.
Every tool returns plain text capped at 6000 characters.

| tool | arguments | what it does |
|---|---|---|
| `literature_search` | `query`, `sources=["semantic_scholar","arxiv","pubmed"]`, `limit=5` | Real API search. Each hit is prefixed with `[source]` and its URL; abstracts are truncated to fit the budget. Retries 429/5xx with backoff (Retry-After honoured). Optional `SEMANTIC_SCHOLAR_API_KEY`, `NCBI_API_KEY`. |
| `math_solve` | `code`, `timeout_s=120` | Runs Python in the sandbox with SymPy, Z3, NumPy, SciPy, NetworkX, pandas, statsmodels (no network, 1 CPU, 1 GB, read-only FS). |
| `stats_test` | `kind`, `data_json` | `ttest`, `paired_t`, `welch`, `mannwhitney`, `cohens_d`, `chi2`, `mcnemar`, `prop_ci`, `multitest` (Holm/BH/Bonferroni). Always reports an effect size and a CI. |
| `causal_analysis` | `csv_path`, `treatment`, `outcome`, `common_causes=[]`, `method="backdoor"`, `num_simulations=20` | DoWhy ATE: estimate, 95% CI (OLS or bootstrap), refutations (permutation placebo, random common cause) and the assumptions. The CSV is mounted read-only. |
| `discover_graph` | `csv_path`, `alpha=0.05` | PC algorithm (causal-learn, Fisher-z). Labelled EXPLORATORY. |
| `lean_check` | `code`, `timeout_s=120` | Type-checks a Lean 4 file in a network-less container: VERIFIED, FAILED (with `Main.lean:line:col` errors), INCOMPLETE (uses `sorry`) or TIMEOUT. `import Mathlib` needs the `meshtrain-lean:mathlib` image; otherwise core Lean (`meshtrain-lean:core`). |
| `workspace_post` | `content`, `kind` (fact, hypothesis, result, question, plan, failure, method), `task_id`, `agent`, `confidence`, `verified`, `sources`, `field` | Wraps `train.memory.workspace.post`; returns the entry id (or "not stored" when the scrubber drops it). |
| `workspace_read` | `task_id`, `query`, `k=8` | Wraps `read`. |
| `memory_search` | `text`, `field`, `k=3` | Wraps `search` (method cards). |
| `analogies` | `text`, `k=3`, `field` | Wraps `analogies` (results from other fields). |
| `workspace_observe` | `ids`, `passed` | Wraps `observe`: outcome feedback for the entries/cards used. |
| `workspace_stats` | (none) | Wraps `stats`. |
| `report_imrad` | `title`, `sections_json`, `audit_trail_json` | Writes an IMRaD markdown report with an audit-trail table to `C:\Users\mariu\ruv\hive\runs\reports\`. Formatter only. |

`task_id` defaults to `CLAUDE_SESSION_ID`, then `"default"`. The workspace tools return `workspace not available: ...`
until `train.memory.workspace` (branch `exp/shared-memory`) can be imported from `MESH_WORKSPACE_ROOT`
(default `C:\Users\mariu\neural-mesh`).

## Environment

| variable | default | meaning |
|---|---|---|
| `MESH_TOOLS_IMAGE` | `meshtrain-grader:2` | sandbox image |
| `MESH_LEAN_IMAGE` | mathlib image if built, else core | Lean image for `lean_check` |
| `MESH_TOOLS_ENABLE` | (all) | comma list of tools to expose, e.g. `math_solve,stats_test` |
| `MESH_WORKSPACE_ROOT` | `C:\Users\mariu\neural-mesh` | where `train.memory.workspace` is imported from |
| `MESH_REPORTS_DIR` | `C:\Users\mariu\ruv\hive\runs\reports` | report output folder |
| `MESH_AGENT_NAME` | `agent` | default `agent` for `workspace_post` |

## Layout

`server.py` (protocol and tool table), `literature.py`, `sandbox_tools.py` (host side of the sandboxed tools),
`sandbox/stats_impl.py` and `sandbox/causal_impl.py` (copied into the container per call), `workspace_tools.py`,
`report.py`, `common.py` (cap, docker runner), `smoke_live.py`, `install.ps1`, `tests/`.

## Tests

    python -m unittest discover -s tools/mesh_mcp/tests -t .     # offline; sandbox tests skip without Docker
    python -m tools.mesh_mcp.smoke_live                           # live: each API once + math/stats via the server
    python -m tools.mesh_mcp.smoke_live --record                  # also refresh tests/fixtures

The grader image is defined in `C:\Users\mariu\ruv\hive\runs\agent-catalog\grader\Dockerfile`
(`requirements-v2.txt` pins every package; `Dockerfile.v1` is the old `:1`). Install steps: see `install.md`.
