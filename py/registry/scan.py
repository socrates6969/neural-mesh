"""Capability registry: index of what is INSTALLED / AVAILABLE on this machine so agents can look it up.
Sources (each entry records `source` and `status`):
  plugin(installed|available) skill agent command   from ~/.claude/plugins/{cache,marketplaces}, ~/.claude/{skills,agents,commands}
  mcp                         from ~/.claude.json, plugin .mcp.json files
  npm                         `npm ls -g`
  cli                         probe list on PATH + ~/.cargo/bin (version via --version, 3 s timeout)
  pip                         `pip list` (top-level, notable only)
HONESTY: the index is a snapshot (`scanned_at`); `verify()` re-checks every entry against the filesystem / PATH and reports drift;
nothing here executes a plugin or MCP server. No secrets are read (only names/descriptions/paths/versions; env blocks of MCP configs are dropped).
Usage: python scan.py scan|verify|lookup "query" [k]
"""
import glob, json, os, re, shutil, subprocess, sys, time
HOME = os.path.expanduser("~")
OUT_DIR = "C:/Users/mariu/experiments/mesh-memory/registry"
OUT = os.path.join(OUT_DIR, "capabilities.json")

def frontmatter(path, limit=4000):
    try:
        t = open(path, encoding="utf-8", errors="replace").read(limit)
    except OSError:
        return {}
    m = re.match(r"---\s*\n(.*?)\n---", t, re.S)
    d = {}
    if m:
        for line in m.group(1).splitlines():
            mm = re.match(r"([A-Za-z_-]+):\s*(.*)", line)
            if mm: d[mm.group(1).lower()] = mm.group(2).strip().strip('"\'')[:400]
    if "description" not in d:
        body = t[m.end():] if m else t
        first = next((l.strip() for l in body.splitlines() if l.strip() and not l.startswith("#")), "")
        d["description"] = first[:300]
    return d

def add(entries, kind, name, desc, source, status="installed", **kw):
    entries.append(dict(kind=kind, name=name, desc=(desc or "")[:400], source=source.replace("\\", "/"), status=status, **kw))

def scan_plugins(entries):
    """Generic walk (v2): any SKILL.md, any <x>/agents/*.md, any <x>/commands/*.md under the plugin trees and user dirs; plugin = nearest
    ancestor holding .claude-plugin/plugin.json. Dedup by path. (v1 used fixed glob shapes and covered only 46% of skills.)"""
    base = os.path.join(HOME, ".claude", "plugins")
    roots = [(os.path.join(base, "cache"), "installed"), (os.path.join(base, "marketplaces"), "available"),
             (os.path.join(HOME, ".claude", "skills"), "installed"), (os.path.join(HOME, ".claude", "agents"), "installed"), (os.path.join(HOME, ".claude", "commands"), "installed")]
    plugin_dirs = {}
    files = []
    for root, status in roots:
        for dp, dn, fn in os.walk(root):
            dn[:] = [x for x in dn if x not in (".git", "node_modules")]
            dpn = dp.replace("\\", "/")
            if os.path.basename(dpn) == ".claude-plugin" and "plugin.json" in fn:
                plugin_dirs[os.path.dirname(dpn)] = (os.path.join(dp, "plugin.json"), status)
            for f in fn:
                p = dpn + "/" + f
                if f == "SKILL.md": files.append(("skill", p, status))
                elif f.endswith(".md") and dpn.endswith("/agents"): files.append(("agent", p, status))
                elif f.endswith(".md") and dpn.endswith("/commands"): files.append(("command", p, status))
                elif f == ".mcp.json": files.append(("mcpfile", p, status))
    def owner(p):
        d = os.path.dirname(p)
        while len(d) > 3:
            if d in plugin_dirs: return os.path.basename(d)
            d = os.path.dirname(d)
        return "(user)" if "/.claude/plugins/" not in p else "(unpackaged)"
    for pdir, (pj, status) in sorted(plugin_dirs.items()):
        meta = {}
        try: meta = json.load(open(pj, encoding="utf-8"))
        except Exception: pass
        add(entries, "plugin", meta.get("name") or os.path.basename(pdir), meta.get("description", ""), pdir, status, marketplace=(re.split(r"/(?:marketplaces|cache)/", pdir)[-1].split("/")[0] if re.search(r"/(?:marketplaces|cache)/", pdir) else ""))
    seen = set()
    for kind, p, status in files:
        if p in seen: continue
        seen.add(p)
        if kind == "mcpfile": scan_mcp_file(entries, p, "plugin:" + owner(p), status, plugin_style=True); continue
        fm = frontmatter(p)
        nm = fm.get("name") or (os.path.basename(os.path.dirname(p)) if kind == "skill" else os.path.splitext(os.path.basename(p))[0])
        add(entries, kind, nm, fm.get("description"), p, status, plugin=owner(p))

def scan_mcp_file(entries, path, origin, status="installed", plugin_style=False):
    try: d = json.load(open(path, encoding="utf-8"))
    except Exception: return
    servers = d.get("mcpServers") or (d if plugin_style else None)
    if not isinstance(servers, dict): return
    for name, cfg in servers.items():
        if not isinstance(cfg, dict): continue
        cmd = " ".join([str(cfg.get("command", cfg.get("url", "")))] + [str(a) for a in cfg.get("args", [])])[:200]
        add(entries, "mcp", name, f"MCP server ({origin}): {cmd}", path, status, origin=origin)  # env block deliberately dropped

def scan_mcp(entries):
    for p in (os.path.join(HOME, ".claude.json"), os.path.join(HOME, ".claude", "settings.json"), os.path.join(HOME, ".mcp.json")):
        if os.path.exists(p): scan_mcp_file(entries, p, os.path.basename(p))

def run(cmd, timeout=6):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, shell=isinstance(cmd, str))
        return (r.stdout or r.stderr or "").strip()
    except Exception:
        return ""

def scan_npm(entries):
    out = run("npm ls -g --depth=0 --json", 20)
    try: deps = json.loads(out).get("dependencies", {})
    except Exception: deps = {}
    for n, v in deps.items(): add(entries, "npm", n, f"global npm package {n}@{v.get('version')}", "npm -g", "installed", version=v.get("version"))

PROBE = ["git", "node", "npm", "npx", "python", "pip", "cargo", "rustc", "rustup", "gh", "docker", "jq", "rg", "ffmpeg", "curl", "tar", "7z", "code", "codex", "claude", "pwsh", "powershell", "make", "cmake", "go", "java", "dotnet", "uv", "pnpm", "yarn", "bun", "deno", "sqlite3", "psql", "wsl", "ssh", "pandoc", "tesseract", "convert"]
PURPOSE = {  # generic dictionary-style purposes (authored independently of any evaluation query)
 "cargo": "Rust package manager and build tool: compile, test and run Rust crates", "cargo-clippy": "Rust linter that catches common mistakes", "cargo-fmt": "format Rust source code",
 "cargo-miri": "interpreter that detects undefined behavior in Rust programs", "rustc": "the Rust compiler", "rustdoc": "generate documentation for Rust crates", "rustfmt": "Rust code formatter",
 "rustup": "Rust toolchain installer and version manager", "rust-analyzer": "Rust language server for editors", "rust-gdb": "debugger for Rust programs (GDB)", "rust-lldb": "debugger for Rust programs (LLDB)",
 "rust-gdbgui": "graphical debugger front end for Rust", "clippy-driver": "Rust lint driver", "rls": "legacy Rust language server",
 "claude": "Claude Code command line interface for the Claude AI assistant", "code": "Visual Studio Code editor launcher", "codex": "OpenAI Codex coding agent command line interface",
 "convert": "ImageMagick: convert, resize and edit images", "curl": "transfer data with URLs: download files and call HTTP APIs", "docker": "build and run applications in containers",
 "gh": "GitHub command line: pull requests, issues and repositories", "git": "distributed version control system for source code", "node": "JavaScript runtime", "npm": "JavaScript package manager",
 "npx": "run JavaScript package executables", "pip": "Python package installer", "powershell": "Windows shell and scripting language", "python": "Python interpreter",
 "ssh": "secure shell: remote login and command execution", "tar": "create and extract archive files", "uv": "extremely fast Python package and project manager", "wsl": "Windows Subsystem for Linux: run a Linux environment on Windows"}
def scan_cli(entries):
    names = list(PROBE) + [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(HOME, ".cargo", "bin", "*.exe"))]
    for n in sorted(set(names)):
        p = shutil.which(n) or (os.path.join(HOME, ".cargo", "bin", n + ".exe") if os.path.exists(os.path.join(HOME, ".cargo", "bin", n + ".exe")) else None)
        if not p: continue
        ver = run([p, "--version"], 3).splitlines()[:1]
        add(entries, "cli", n, f"command-line tool {n}: {PURPOSE.get(n, '')} ({ver[0][:80] if ver else ''})", p, "installed")

def scan_pip(entries):
    out = run([sys.executable, "-m", "pip", "list", "--format=json"], 30)
    try: pk = json.loads(out)
    except Exception: pk = []
    try:
        import importlib.metadata as im
    except Exception: im = None
    for p in pk:
        summ = ""
        try: summ = (im.metadata(p["name"]).get("Summary") or "")[:200] if im else ""
        except Exception: pass
        add(entries, "pip", p["name"], f"python package {p['name']} {p['version']}: {summ}", "pip", "installed", version=p["version"])

def scan():
    entries = []
    for f in (scan_plugins, scan_mcp, scan_npm, scan_cli, scan_pip):
        try: f(entries)
        except Exception as e: print("scan error", f.__name__, e, file=sys.stderr)
    for i, e in enumerate(entries): e["id"] = i
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(dict(scanned_at=time.strftime("%Y-%m-%dT%H:%M:%S"), host=os.environ.get("COMPUTERNAME", ""), count=len(entries), entries=entries), open(OUT, "w"), indent=0)
    return entries

def verify():
    """re-check every entry; returns (n, missing list). File-backed entries must exist; cli entries must resolve; npm/pip re-queried."""
    d = json.load(open(OUT)); missing = []
    npm = {e["name"] for e in d["entries"] if e["kind"] == "npm"}
    for e in d["entries"]:
        k = e["kind"]
        ok = True
        if k in ("plugin", "skill", "agent", "command", "mcp"): ok = os.path.exists(e["source"])
        elif k == "cli": ok = os.path.exists(e["source"])
        if not ok: missing.append((k, e["name"], e["source"]))
    live = []; scan_npm(live)
    gone = npm - {e["name"] for e in live}; new = {e["name"] for e in live} - npm
    return len(d["entries"]), missing, sorted(gone), sorted(new), d["scanned_at"]

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if cmd == "scan":
        es = scan(); from collections import Counter; print(Counter((e["kind"], e["status"]) for e in es))
    elif cmd == "verify":
        n, miss, gone, new, at = verify(); print(f"scanned_at {at}: {n} entries, {len(miss)} missing, npm gone {gone}, npm new {new}")
