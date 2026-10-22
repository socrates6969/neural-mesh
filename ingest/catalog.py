"""Agent catalog: every agent definition installed for Claude Code, so the mesh can see agents it has never dispatched.

Sources (read-only):
  * built-ins (general-purpose, Explore, Plan, ...)
  * user agents      ~/.claude/agents/**/*.md            -> name
  * plugin agents    <installPath>/agents/**/*.md        -> "<plugin>:<name>"  (installed_plugins.json, latest entry per plugin)
  (plugins and synced plugins come from `claude plugin list --json`, which also says whether each one is enabled)
Only name / description / tools / model / source are kept; agent bodies are never copied.
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

HOME = Path(os.environ.get("CATALOG_HOME", str(Path.home())))
BUILTINS = {
    "general-purpose": "General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks.",
    "Explore": "Read-only search agent for broad fan-out searches across many files.",
    "Plan": "Software architect agent for designing implementation plans.",
    "claude-code-guide": "Questions about Claude Code, the Agent SDK and the Claude API.",
    "statusline-setup": "Configure the Claude Code status line.",
}
_FM = re.compile(r"^---\s*\n(.*?)\n---", re.S)


def _frontmatter(text: str) -> dict:
    m = _FM.match(text.lstrip("﻿"))
    if not m:
        return {}
    out, key = {}, None
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if km and not line.startswith((" ", "\t")):
            key, val = km.group(1), km.group(2).strip()
            out[key] = val.strip("\"'") if val not in ("|", ">", "|-", ">-") else ""
        elif key and line.startswith((" ", "\t")) and key == "description":
            out[key] = (out.get(key, "") + " " + line.strip()).strip()
    return out


def _agent_files(root: Path):
    d = root / "agents"
    return sorted(Path(p) for p in glob.glob(str(d / "**" / "*.md"), recursive=True)) if d.is_dir() else []


def _entry(path: Path, prefix: str | None, source: str, plugin: str | None, enabled: bool, root: Path | None = None):
    try:
        fm = _frontmatter(path.read_text(encoding="utf-8", errors="replace")[:20000])
    except OSError:
        return None
    # no frontmatter name: Claude Code names the agent by its path under agents/, ":"-joined (searchfit-seo:seo-auditor:AGENT)
    rel = ":".join(path.relative_to(root / "agents").with_suffix("").parts) if root else path.stem
    base = (fm.get("name") or rel).strip()
    if not base or base.upper() in ("README", "CLAUDE", "AGENTS") and not fm.get("description"):
        return None
    desc = re.sub(r"\s+", " ", fm.get("description", "")).strip() or (f"Agent from {plugin.split('@')[0]} plugin" if plugin else "")
    if not desc:
        return None
    return {"name": f"{prefix}:{base}" if prefix else base, "description": desc[:1200], "tools": fm.get("tools", ""),
            "model": fm.get("model", ""), "source": source, "plugin": plugin, "enabled": enabled, "path": str(path)}


def _plugin_list(plugins_dir: Path, enabled_map: dict) -> list[dict]:
    """Installed plugins with installPath + enabled. Authoritative source: `claude plugin list --json` (what Claude Code itself loads);
    fallback when the CLI is unavailable: installed_plugins.json (latest entry per plugin) + enabledPlugins."""
    import shutil
    import subprocess
    exe = shutil.which("claude")
    if exe and not os.environ.get("CATALOG_NO_CLI"):
        try:
            r = subprocess.run([exe, "plugin", "list", "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            data = json.loads(r.stdout)
            if isinstance(data, list) and data:
                best = {}
                for pl in data:   # one row per id: prefer an enabled user-scope entry
                    k = pl.get("id")
                    if not k:
                        continue
                    rank = (bool(pl.get("enabled")), pl.get("scope") == "user")
                    if k not in best or rank > best[k][0]:
                        best[k] = (rank, pl)
                return [v[1] for v in best.values()]
        except Exception:
            pass
    try:
        inst = json.loads((plugins_dir / "installed_plugins.json").read_text(encoding="utf-8")).get("plugins", {})
    except Exception:
        inst = {}
    return [{"id": k, "installPath": v[-1].get("installPath", ""), "enabled": bool(enabled_map.get(k, False))} for k, v in inst.items() if v]


def build(home: Path = HOME) -> list[dict]:
    claude = home / ".claude"
    plugins_dir = claude / "plugins"
    try:
        enabled_map = json.loads((claude / "settings.json").read_text(encoding="utf-8")).get("enabledPlugins", {})
    except Exception:
        enabled_map = {}
    rows = [{"name": n, "description": d, "tools": "*", "model": "", "source": "builtin", "plugin": None, "enabled": True, "path": ""}
            for n, d in BUILTINS.items()]
    for p in _agent_files(claude):
        e = _entry(p, None, "user", None, True, claude)
        if e:
            rows.append(e)
    for pl in _plugin_list(plugins_dir, enabled_map):
        name = pl["id"].split("@")[0]
        for p in _agent_files(Path(pl.get("installPath") or "")):
            e = _entry(p, name, "synced" if pl["id"].endswith("@synced") else "plugin", pl["id"], bool(pl.get("enabled")), Path(pl.get("installPath") or ""))
            if e:
                rows.append(e)
    seen, out = {}, []
    for r in rows:   # first definition wins, but an enabled copy beats a disabled one
        k = r["name"]
        if k in seen:
            if r["enabled"] and not out[seen[k]]["enabled"]:
                out[seen[k]] = r
            continue
        seen[k] = len(out)
        out.append(r)
    return sorted(out, key=lambda r: r["name"].lower())


def write(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"version": 1, "agents": rows}, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, path)
    return {"total": len(rows), "enabled": sum(r["enabled"] for r in rows), "path": str(path)}


if __name__ == "__main__":
    import sys
    rows = build()
    print(json.dumps(write(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent_catalog.json"), rows)))
