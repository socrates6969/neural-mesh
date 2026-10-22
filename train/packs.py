"""Knowledge packs: per (agent, field) playbooks the mesh hands to an agent when it dispatches it.

The pack writer is a separate lean Claude run that MAY use the web (open sources, docs, APIs) and sees only TUNE
tasks: the agent's failures with the correct answer and reference solution, plus a few successes. It never sees
held-out tasks. A pack must be general method, not a list of answers (enforced by a length cap and a leak check).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .runner import CLAUDE

MAX_WORDS = 900

WRITER_PROMPT = """You are improving a specialist AI agent at {field}. You write its KNOWLEDGE PACK: a compact playbook that is
given to the agent together with each new {field} task. The agent was just tested on practice tasks.

Agent: {agent}
What the agent is told about itself: {desc}

Current pack (may be empty):
<<<
{pack}
>>>

Practice results: {n_ok} solved, {n_bad} failed. The failures, each with the agent's answer and the correct answer:
{failures}

Write the improved pack. Requirements:
- Diagnose WHY the failures happened (method errors, missed cases, arithmetic slips, format, not verifying) and give
  general, reusable rules that prevent that class of error. Do not mention or solve the specific practice problems,
  and never include their answers.
- Include a short verification checklist and concrete tool habits (e.g. "check the result with a 10-line Python
  brute force or numeric substitution before answering"; only numpy and the Python standard library are installed).
- You may research open sources (textbooks, standard references, official docs, open datasets/APIs) with web
  search to add the most valuable techniques for this field; cite at most 5 sources by name/URL at the end.
- Keep what in the current pack still helps; remove what does not. Maximum {max_words} words. Plain text/markdown.
Output ONLY the pack, starting with the line "# {field} pack for {agent}"."""


def _fmt_failures(fails: list[dict], limit: int = 12) -> str:
    out = []
    for f in fails[:limit]:
        t, rec = f["task"], f["rec"]
        gold = t["answer"] if t["kind"] != "code" else "(hidden unit tests; the reference solution is shown)"
        ref = (t.get("ref") or "")[:1200]
        out.append("### Task\n%s\n### Agent output (end)\n%s\n### Correct answer\n%s\n%s" % (
            t["prompt"][:1800], (rec.get("text") or rec.get("error") or "")[-1200:], gold, ("### Reference solution\n" + ref) if ref else ""))
    return "\n\n".join(out) or "(none)"


def leaks(pack: str, tune_tasks: list[dict]) -> list[str]:
    """Long verbatim chunks of task statements inside a pack = memorisation, not method."""
    bad = []
    for t in tune_tasks:
        words = re.findall(r"\w+", t["prompt"])
        for i in range(0, max(0, len(words) - 12), 6):
            chunk = " ".join(words[i:i + 12])
            if chunk and chunk in " ".join(re.findall(r"\w+", pack)):
                bad.append(t["id"])
                break
    return bad


def write_pack(agent: dict, field: str, pack: str, fails: list[dict], n_ok: int, timeout: float = 1200,
               budget_usd: float = 4.0) -> tuple[str, float]:
    prompt = WRITER_PROMPT.format(field=field, agent=agent["name"], desc=agent["description"], pack=pack or "(empty)",
                                  n_ok=n_ok, n_bad=len(fails), failures=_fmt_failures(fails), max_words=MAX_WORDS)
    tmp = Path(tempfile.mkdtemp(prefix="mt-writer-"))
    (tmp / "mcp.json").write_text('{"mcpServers":{}}', encoding="utf-8")
    cmd = [CLAUDE, "-p", "--no-session-persistence", "--setting-sources", "project", "--strict-mcp-config",
           "--mcp-config", str(tmp / "mcp.json"), "--disable-slash-commands", "--tools", "WebSearch,WebFetch,Bash",
           "--allowedTools", "WebSearch", "WebFetch", "Bash(python:*)", "--output-format", "json", "--max-turns", "25",
           "--max-budget-usd", str(budget_usd)]
    try:
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=str(tmp))
        d = json.loads(r.stdout)
        text, cost = (d.get("result") or "").strip(), float(d.get("total_cost_usd") or 0)
    except Exception:
        text, cost = "", 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    i = text.find("# ")
    text = text[i:] if i >= 0 else text
    words = text.split()
    if len(words) > MAX_WORDS * 1.15:
        text = " ".join(words[: int(MAX_WORDS * 1.15)])
    return text, cost
