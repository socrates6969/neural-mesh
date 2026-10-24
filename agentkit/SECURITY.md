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
- `agentkit.approve` refuses when stdin/stdout are not a terminal; when it runs inside an agent or Claude Code process
  (`CLAUDECODE`, `CLAUDE_CODE_*`, `CLAUDE_SESSION_ID`, `CLAUDE_PID`, or an ancestor that is `claude`/`claude.exe`, a
  node/bun/deno process running a claude CLI, or any ancestor with "claude" in its command line); and when the
  parent-process chain cannot be read or ends early. It uses Windows Hello when it is configured, otherwise a random
  per-run code typed back. The entry is shown with control characters and ANSI escape sequences removed (with a
  warning if any were).

## What is NOT protected

**An agent that can already run arbitrary code as you is not contained.** All agents run as the same Windows user
with shell access. Such an agent can read both key files, compute valid signatures, write the workspace database
directly, or edit these scripts. The checks above raise the cost and stop text-only attacks; they are not a boundary
against code running as you. Windows Hello and the parent-process checks only make an approval harder to fake from
inside an agent session; they do not stop code that can drive the desktop or start processes outside the session.

Full protection would need the keys and the signing step in a separate OS account or service (or hardware-backed
keys) that agents cannot access, with approvals taken through that service. That is not set up.

## Mitigations in place

| Risk | Mitigation |
|---|---|
| Agent output claims "verified" | Always stored unverified; the claim is display-only (`claimed_verified`). |
| Forged pipeline or card provenance in text | `pipeline_sig` required; unsigned `pipeline:*` and cards are untrusted across sessions. |
| Forged or replayed human approval | `approval_sig` over id, task, agent, kind, content and verifier; random code or Windows Hello; refused inside agent processes. |
| Tampering with stored rows | Signatures cover the stored content; an edit breaks them. |
| Prompt injection through injected memory | Evidence-only framing, marker and fence stripping, quoted and capped cards, provenance on every item, other sessions only when signed, the 900-token cap. |
| Terminal tricks in approval display | C0/C1 control characters and ANSI/ESC sequences removed, with a warning. |
| Key theft by other local users | Keys created atomically (O_EXCL), owner-only ACL, deleted if the ACL cannot be set. |
| Side effects without consent | Protocol consequence check: no external action without explicit approval (`needs_human`). |

Switches: `NEURAL_MESH_CROSS_SESSION=off` (no other-session items), `NEURAL_MESH_WORKSPACE=off`,
`NEURAL_MESH_CAPTURE=off`, `NEURAL_MESH_ASSIST=off`.