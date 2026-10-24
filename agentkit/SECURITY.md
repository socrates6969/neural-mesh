# Agent kit: security model

## What is protected

**Hostile text cannot forge trust.** Prompt injection in web pages, documents, tool output or other agents' replies
cannot make an entry verified, approve it, or get it shown to agents in other sessions as trusted. Concretely:

- `verified` in model output is never stored as verified. The capture hook stores every post `verified=False` and keeps
  the agent's claim only as `claimed_verified`, for display.
- An entry crosses a session boundary only if it carries a valid signature:
  - `pipeline:grader` entries and grader method cards need `pipeline_sig`, an HMAC-SHA256 keyed with
    `~/.claude/mesh-pipeline.key`. Only the trusted writers sign: `train.memory.ingest_training` and the card builder.
  - `human:<user>` approvals need `approval_sig`, an HMAC-SHA256 keyed with `~/.claude/mesh-approve.key`. Only
    `python -m agentkit.approve` signs, after a human-presence check.
  - Both signatures cover the stored content, so an edited entry or a copied signature does not verify. Missing or
    malformed keys fail closed: nothing verifies.
- Injected memory is framed as background evidence, never instructions. It is cleaned of code fences and of anything
  that looks like our markers, cards are quoted line by line and capped, and every item shows its provenance
  `[session / agent / verified-by]`. Other sessions' raw output and failures are never injected.
- **Human approval rests on Windows Hello, which is mandatory.** `agentkit.approve` shows the entry id and content
  first (control characters and ANSI escape sequences removed, with a warning if any were), then requires a
  successful Windows Hello verification (UserConsentVerifier) AND a random per-run code typed back. Hello not
  configured, unavailable, an error, a cancel or a timeout all refuse; there is no code-only fallback. On POSIX there
  is no Hello, so approval always refuses.
- The process checks are **defense in depth, not proof of a human**. Approve also refuses when stdin/stdout are not a
  terminal; inside an agent or Claude Code process (`CLAUDECODE`, `CLAUDE_CODE_*`, `CLAUDE_SESSION_ID`, `CLAUDE_PID`,
  or an ancestor that is `claude`/`claude.exe`, node/bun/deno running a claude CLI, or any ancestor with "claude" in
  its command line); and when the parent chain is inconclusive. The trust root comes from the OS, not the
  environment: the process that owns the desktop shell window (GetShellWindow), whose image is
  `<GetSystemWindowsDirectoryW>\explorer.exe` and whose session is the approver's session (SystemRoot variables are
  ignored). Below the shell, a missing parent, a loop, an ancestor in another session or a launch through WmiPrvSE,
  svchost or taskhostw refuses; above it the walk continues to pid 0/4 where readable, still checking for agents. On
  POSIX, ppid 1 or a subreaper launch is inconclusive, so approval refuses.
- Items that cross a session boundary are displayed and gated on signed fields only (id, kind, task, agent,
  verified_by, content; for cards id, field, text). Unsigned columns such as confidence, an entry's field, sources or
  relevance can rank items but are never shown as facts.
- The pipeline MAC includes the entry id (signed by the writer inside the insert transaction), so a signature does
  not verify on a copied row.

## What is NOT protected

**An agent that can already run arbitrary code as you is not contained.** All agents run as the same Windows user
with shell access. Such an agent can read both key files, compute valid signatures, write the workspace database
directly, or edit these scripts. The checks above raise the cost and stop text-only attacks; they are not a boundary
against code running as you. Windows Hello requires the person at the machine (face, fingerprint or PIN), which an agent cannot supply by itself;
the parent-process checks only make an approval harder to fake from inside an agent session. Neither stops code
that can read the keys and write the database directly.

Full protection would need the keys and the signing step in a separate OS account or service (or hardware-backed
keys) that agents cannot access, with approvals taken through that service. That is not set up.

Two limits of the approval check, stated plainly:

- **Ancestry is a hint, not proof.** The parent-process checks catch an approval run from inside an agent or Claude
  Code session by accident or by a text-only attack. Code running as you can start a process outside that tree.
- **The Hello result is trusted only as far as the process environment is.** `approve.py` runs Windows PowerShell
  from its pinned absolute path (`<GetSystemWindowsDirectoryW>\System32\WindowsPowerShell\v1.0\powershell.exe`)
  with a minimal clean environment (SystemRoot, PATH = System32, no PSModulePath) and cwd = System32, then reads its
  output. Code running as you could still tamper with that process or with `approve.py` itself and fake the output.
- **Upgrade path if the threat model expands:** replace the Hello yes/no with a nonce-bound signature from a
  Windows Hello key (`Windows.Security.Credentials.KeyCredentialManager`: sign a per-approval nonce, verify the
  signature against the registered public key). The approval would then be checked cryptographically instead of
  trusting a process's output.

## Mitigations in place

| Risk | Mitigation |
|---|---|
| Agent output claims "verified" | Always stored unverified; the claim is display-only (`claimed_verified`). |
| Forged pipeline or card provenance in text | `pipeline_sig` required; unsigned `pipeline:*` and cards are untrusted across sessions. |
| Forged or replayed human approval | `approval_sig` over id, task, agent, kind, content and verifier; mandatory Windows Hello plus a random typed code; process checks (agent env, ancestry to the OS shell) as defense in depth. |
| Tampering with stored rows | Signatures cover the stored content and the row id; an edit or a copied row fails verification. |
| Prompt injection through injected memory | Evidence-only framing, marker and fence stripping, quoted and capped cards, provenance on every item, other sessions only when signed, the 900-token cap. |
| Terminal tricks in approval display | C0/C1 control characters and ANSI/ESC sequences removed, with a warning. |
| Key theft by other local users | Keys created atomically (O_EXCL), owner-only ACL, deleted if the ACL cannot be set. |
| Side effects without consent | Protocol consequence check: no external action without explicit approval (`needs_human`). |

Switches: `NEURAL_MESH_CROSS_SESSION=off` (no other-session items), `NEURAL_MESH_WORKSPACE=off`,
`NEURAL_MESH_CAPTURE=off`, `NEURAL_MESH_ASSIST=off`.