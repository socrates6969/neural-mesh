## Finding code and docs: codemem first

When a task needs you to locate code, a definition, a config value, a migration or a doc section, make your FIRST command one query,
before any grep / find / cat / git log:

    codemem query "<what you are looking for; add identifiers or paths if you know them>" --k 5

It prints file:line pointers with 1-2 snippet lines from a BM25 index of this repo (git-tracked and untracked non-ignored files;
secrets, .env, keys and data dirs are excluded; refreshed nightly). Use `--repo <name>` when working outside the repo.

- Pointers are leads, not proof: Read the lines before you rely on them or edit.
- If the top hits look unrelated, say so and fall back to grep; do not repeat the query with the same words.
- After large edits the index can be stale; `codemem update` refreshes the current repo in about a second.
