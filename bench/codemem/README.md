# codemem: BM25 project memory for code and docs (agents find file:line in one step)

Source: `src/codemem/` (lib, also `neural-mesh codemem ...`) and `src/bin/codemem.rs` (small stand-alone exe, same source via `#[path]`, 3 MB vs 15 MB:
process start is the dominant hook cost on this box, hello-world exe = 60 ms p50). Tests: `cargo test --release --offline --lib codemem` (20) and
`python -m unittest hooks.tests.test_codemem_hook` (8, through the real exe).

## What it is
* Corpus: neural-mesh, hebbian-router, freight-recovery, freight-recovery-build, freight-recovery-brand (+ agent run notes under
  `~/.claude/plugins/data/statsclaw-synced/workspace/freight-recovery/runs`, indexed by absolute path). Listing = `git ls-files -z -co --exclude-standard`
  (so .gitignore is honoured by git), then an allow-list of extensions, minus secrets/secret/data/node_modules/target/dist/vendor*/dot dirs (except
  .github, .claude), `.env*`, `*.key/pem/p12/pfx/jks`, credentials*, `*.tfvars/tfstate`, `settings.local.json`, lockfiles, files > 400 KB (json > 200 KB), binaries.
  neural-mesh additionally excludes `bench/battery*`, `bench/results|work|data|tests`, `bench/PREREG.json`, `bench/codemem/`, `results/` (answer keys: a retrieval
  benchmark must not read its own answers).
* Chunks: by symbol (Rust/Python/TS/SQL/Prisma/Terraform/shell, doc comments and attributes attached) or heading (markdown, fences respected), <= 70 lines
  and 5 KB, each with its 1-based line range. Every chunk text goes through `src/codemem/scrub.rs` (port of ingest/scrub.py scrub-v1: keys, JWT, private
  keys, bearer, connection strings, password assignments with the same placeholder rules, Luhn cards, high-entropy tokens, email, phone) BEFORE it is
  tokenised or stored; matches are redacted in place (the Python policy drops the record; for code that would lose the pointer). Deviation: sensitive-path
  MENTIONS are not redacted (sensitive FILES are never read). Counts per category are reported by `update`, never the matches.
* Tokens: whole identifiers `[A-Za-z0-9_]+` lowercased + snake_case/camelCase parts (`split`, ON; see "Decisions"), title x2, path tokens, and a `def:<name>`
  token for every declaration line (`pub const X`, `fn f`, `class C`, `NAME = ...`). BM25 k1 .9, b .5 (RD-LOG R54). Query: backticked spans weigh 2, path-like words
  1.5 (and boost chunks of matching files x1.5), identifier-like words add `def:` terms (weight 4); prose chunks (md/txt/run notes) are multiplied by .65.
* Index: `data/codemem/<repo>/index.cmem` (one file, atomic replace, version 2; an older version is rebuilt, never patched). Per file mtime+size, then FNV hash;
  unchanged files are copied (term ids remapped), only changed files are re-chunked/scrubbed/tokenised.
* CLI: `codemem query "<q>" [--repo X|--cwd D] [--k 5] [--json]`, `update [--repo X|--all] [--force]`, `status`, `hook`, `inject`, `gate`, `bench`, `eval`, `add NAME PATH`.
* Hook (`codemem hook`, UserPromptSubmit): cheap lexical gate (paths, backticks, identifiers, `::`/`()`/flags, code nouns, "where/which ... defined/enforced/..." forms, repo
  terms; slash commands and < 3 words stop) -> repo = longest configured path prefix of `cwd` -> BM25 -> relevance floor (coverage >= .20, >= 1 rare term, >= 2 matched
  terms) -> <= 5 pointers, 1-2 snippet lines each, <= 900 chars (~250 tokens at 3.5 chars/token; measured mean 196 tokens by chars/4, max 287 by chars/3), else nothing.
  Exit 0 on every path (bad JSON, no repo, no index, corrupt index, 100 ms internal budget `NEURAL_MESH_CODEMEM_BUDGET_MS`, panics). `NEURAL_MESH_CODEMEM=off` disables.
  Log line per call (no prompt text) in `data/codemem/hook.jsonl` (`NEURAL_MESH_CODEMEM_LOG=off` to skip).

## Measurements (every number: command, n; machine quiet unless noted)
**Speed.** In-process query (open + load index + parse + search + 5 pointers; `codemem bench --repo freight-recovery-build`, n=200): p50 .56 ms, p95 .67 ms, max .83 ms;
neural-mesh (13.8k chunks, n=300): p50 .87, p95 1.25 ms. End to end as a process (python `subprocess`, n=60): `codemem query` p50 44 / p95 46 ms; hook p50 45 / p95 50 ms
(fires) and 44 / 48 (silent); hello-world exe floor 60 / 62 ms on the same box and moment. The 3.2 MB exe is why the hook beats the 15 MB `neural-mesh.exe` (112 / 125 ms).
Targets (query p95 <= 50 ms warm, hook <= 150 ms): met. Incremental update, nothing changed, all 5 repos: 0.37 s wall; one changed file: test asserts < 5 s. Cold full build of all
five repos: 3.5 s (neural-mesh 1.3 s, 13.7k chunks, 11 MB). Under load (other agents building) the same hook measured p95 66 ms.

**(iii) Fire rate on real user prompts** (`python bench/codemem/measure_fire.py`, 241 real main-session prompts, same loader as hooks/measure_recall.py, tokens = chars/4):
gate passes 29%; hook fires **1/241 = 0.4%** with the prompt's recorded cwd (~191 tokens when it fires, 0.8 tokens/prompt on average); pretending every prompt was typed in the repo
(freight-* sessions -> freight-recovery-build, others -> neural-mesh) 11/241 = 4.6% (9.0 tokens/prompt), and most of those are pasted plugin-install text: real prompts are directives, as
loop-closer found for the recall hook. Exit code non-zero: 0 of 482.

**(i) bench battery** (`python bench/codemem/run_battery.py --split tune --cond A,B --tag cmN --k 1 --workers 3 --sandbox job`, haiku via `claude -p`, n=120, TUNE split = descriptive,
2 runs): B = real hook output prepended. Run 1 (cm1, before the `def:` fix): research_private 0.00 -> 0.80 (3 of 15 missed: the hook ranked usages above the declaration line); run 2 (cm2, shipped
version): **research_private 0.000 -> 1.000 [1.000, 1.000]** (>= v2's 1.00: met), ALL 0.775 -> 0.917 +0.142 [+0.075, +0.208]. Public items (n=105): B - A = +0.019, 90% CI [-0.010, +0.048]
in both runs; **TOST +-.03 NOT met** (the upper bound is a possible gain beyond the margin; at n=105, K=1 the interval cannot be narrower; non-inferiority holds, lower bound -0.010 > -0.03; A rerun noise
on the same items is -0.010 [-0.048, +0.029]). Tokens +8 [-40, +47] per item, wall -0.3 s [-0.8, +0.2] (cm2). bench/frontend.py was not edited (its sha is the pre-registration key).

**(ii) agent-task A/B** (`python bench/codemem/agent_ab.py run --tag main2 --split all --arms A,B --reps 2 --workers 3`; `report --tag main2 --split held`): 47 questions about the user's
repos, topics mined from past transcripts (see build_questions.py: sentences that cite `path:line` for a claim about the code; wording and key written by hand from the repo), 16 tune / 31 held
(i % 3). `claude -p --model haiku --setting-sources project --strict-mcp-config`, tools Read/Grep/Glob **plus read-only Bash** (grep, rg, find, cat, ls, head, tail, wc, sed, git grep, git ls-files),
answer `ANSWER: path:line`, graded file + line within the key range +-5, usage fields from stream-json, 2 reps, paired bootstrap 95% CIs over items. A = question only; B = the REAL hook via `--settings`.
Final (hook v2, tag main2), held n=31: accuracy (file+line) +0.03 [-0.06, +0.13] (no loss); fresh tokens -29% (-2.2k [-4.0k, -0.6k]); all tokens incl. cache reads -36%; tool calls -2.66 [-3.74, -1.74] (-57%);
turns -1.9; wall (API) -32% (-5.4 s [-8.5, -2.3]); cost -37% (-$0.013 [-0.019, -0.007]). All 47: fresh tokens -33%, time -33%, calls -56%, accuracy +0.04 [-0.04, +0.13]. Baseline A found the right
file+line on 88% (held 86%). Earlier hook version v1 (tag main1, before the `def:` fix; also arm C): held n=31 tokens -1% (n.s.), calls -45% *, time -9% (n.s.); all 47 time -19% *, cost -24% *, accuracy +0.01.
v1 -> v2 changed retrieval only through the declaration boost; the main1/main2 difference is partly rerun noise of the baseline (A wall 14.9 s vs 16.8 s). Arm C (CLAUDE.md snippet + `codemem` allowed in Bash, no hook),
v1 index, held n=31: used codemem on 92% of runs, fresh tokens -19% (n.s.), time +8% (n.s.), accuracy -0.06 [-0.18, +0.02] (n.s.), Grep/Glob calls -1.8 *: the snippet changes the tool mix but did not beat the baseline here.
The queen's baseline says agents search mostly via Bash (median 24 search-like Bash calls per dispatch); in this isolated test the baseline used Grep/Glob (2.6) and almost no Bash search (0.2), so the number of
calls to save is much smaller than in real dispatches, and the A/B does not measure that regime.

## Decisions (and what contradicts RD-LOG)
* `split` (snake/camel parts) ON: RD-LOG R18/R55 rejected stemming and camelCase splitting on Rust-crate DOC text. On code questions (retrieval only, no LLM, `codemem eval`): tune file hit@5 .75 -> .875,
  hit@1 .375 -> .56; held file hit@5 .81 -> .90, hit@1 .48 -> .52. Kept for code; docs-only corpora should keep it off (`"split": false` in data/codemem/config.json, index rebuilds).
* prose factor .65 (tune hit@1 .19 -> .375 at .8), `def:` weight 4 (private-repo battery 15/15 definition lines visible vs 14/15 at 2; tune hit@5 .875 -> .81 (one item), held unchanged).
* The gate and floor were set on the 16 tune questions only; held was used once to confirm (split vs whole, docf).

## Limits (honest)
* The question set is mine (topics from transcripts, wording and keys by hand), one model (haiku), one baseline harness; tune questions were also in the 47-item analysis. Retrieval hit-rate on held (file in top 5) is
  .90, line-in-chunk .74, i.e. 1 in 10 injected hint sets miss the file (the agent can still grep).
* The hook only sees main-session user prompts (UserPromptSubmit does not fire for subagent dispatch prompts), and real prompts rarely pass (0.4%); the agent-facing route is the CLI plus `snippet.md`, whose A/B was not positive.
* The index goes stale after edits until `codemem update` (nightly, or by hand, ~0.4 s); pointers are leads, a wrong line costs one Read.
* `target/release/neural-mesh.exe` was rebuilt once while building (HEAD plus the then-uncommitted edits of other agents).

## Install (nothing is installed by this change)
1. `cargo rustc --release --offline --bin codemem -- -C strip=symbols` (WinLibs on PATH); `copy target\release\codemem.exe data\codemem\bin\codemem.exe`; `data\codemem\bin\codemem.exe update`.
2. Hook: merge `hooks/settings-hooks-codemem.json` into `~/.claude/settings.json` (the queen owns that file). Off switch: `NEURAL_MESH_CODEMEM=off`.
3. Nightly: add before `Write-Log "=== nightly done` in llm/scripts/run_nightly.ps1 (never changes the exit code):
   ```
   # 4) codemem: incremental code/doc index refresh for agents (data\codemem; ~0.4 s when nothing changed)
   $cm = Join-Path $repo 'data\codemem\bin\codemem.exe'
   if (Test-Path -LiteralPath $cm) { $crc = Invoke-Logged $cm @('update') $Log; Write-Log "codemem update exit $crc" }
   ```
4. Agents: paste `bench/codemem/snippet.md` into a project CLAUDE.md (text only; put `data\codemem\bin` on PATH or call the exe by full path).
