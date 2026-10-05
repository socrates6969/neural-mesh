"""Independent completeness audit of the capability registry: enumerate SKILL.md / agents / commands by a DIFFERENT method
(os.walk over the plugin trees) and compare with the index. Prints missing counts and the directory-shape of the misses."""
import collections, json, os, re
HOME = os.path.expanduser("~").replace("\\", "/")
d = json.load(open("C:/Users/mariu/experiments/mesh-memory/registry/capabilities.json"))
idx = collections.defaultdict(set)
for e in d["entries"]: idx[e["kind"]].add(e["source"])
roots = [HOME + "/.claude/plugins/cache", HOME + "/.claude/plugins/marketplaces", HOME + "/.claude/skills", HOME + "/.claude/agents", HOME + "/.claude/commands"]
import subprocess
found = collections.defaultdict(set)
def find(r, args):  # independent enumeration with GNU find (different code path from scan.py's os.walk)
    out = subprocess.run(["C:/Program Files/Git/usr/bin/find.exe", r] + args, capture_output=True, text=True).stdout.splitlines()
    return {l.strip().replace("\\", "/") for l in out if l.strip()}
for r in roots:
    if not os.path.isdir(r): continue
    found["skill"] |= find(r, ["-name", "SKILL.md", "-not", "-path", "*/.git/*", "-not", "-path", "*/node_modules/*"])
    found["agent"] |= find(r, ["-path", "*/agents/*.md", "-not", "-path", "*/.git/*", "-not", "-path", "*/node_modules/*", "-type", "f"])
    found["command"] |= find(r, ["-path", "*/commands/*.md", "-not", "-path", "*/.git/*", "-not", "-path", "*/node_modules/*", "-type", "f"])
# find's -path */agents/*.md also matches nested deeper files; keep only files whose parent dir is named agents/commands
found["agent"] = {p for p in found["agent"] if p.rsplit("/", 2)[-2] == "agents"}
found["command"] = {p for p in found["command"] if p.rsplit("/", 2)[-2] == "commands"}
for k in ("skill", "agent", "command"):
    miss = sorted(found[k] - idx[k])
    shapes = collections.Counter(re.sub(r"[^/]+", "*", m.replace(HOME + "/.claude/plugins/", "")) for m in miss)
    print(k, "independent", len(found[k]), "indexed", len(idx[k]), "missing", len(miss), shapes.most_common(4))
    print("   e.g.", miss[:2])
