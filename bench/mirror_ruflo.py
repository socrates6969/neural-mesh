"""bench/mirror_ruflo.py - mirror the mesh's credited memories and outcome summaries into ruflo memory,
namespace "vector-patterns", so ruflo agents can query them (memory_search / memory_retrieve).
Does not duplicate the ingestor: it only READS data/lifelong/memory/items.jsonl and data/lifelong/outcomes.jsonl
(both already scrubbed by ingest.scrub) and re-applies a cheap secret guard before anything leaves this process.

  python bench/mirror_ruflo.py --dry-run              # write bench/work/ruflo_mirror_batch.jsonl only
  python bench/mirror_ruflo.py --apply --db <path>    # push with `ruflo memory store --path <db>` (idempotent upsert)
Live MCP route (when the AgentDB bridge is down): feed the batch file to the memory_store tool instead.
Only memories with status=active and value >= --min-value are mirrored; keys are the ingestor ids (mem:<sha16>)."""
import argparse, json, pathlib, re, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
LIFE = HERE.parent / "data" / "lifelong"
CLI = "C:/Users/mariu/AppData/Local/npm-cache/_npx/85fb20e3e7e3a233/node_modules/@claude-flow/cli/bin/cli.js"
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{15,}\.|ghp_[A-Za-z0-9]{20,}|password\s*[:=]|postgres(ql)?://[^ ]*:[^ ]*@|-----BEGIN)", re.I)
NS = "vector-patterns"

def batch(min_value):
    out = []
    p = LIFE / "memory" / "items.jsonl"
    if p.exists():
        for l in open(p, encoding="utf8"):
            it = json.loads(l)
            if it.get("status") != "active" or it.get("value", 0) < min_value or SECRET.search(it["text"]): continue
            out.append(dict(key=it["id"], namespace=NS, tags=["memory", it.get("tag", ""), "credited"],
                            value=json.dumps(dict(text=it["text"], tag=it.get("tag"), project=it.get("project"), value=it.get("value"), retrievals=it.get("retrievals"), credit=it.get("credit")))))
    p = LIFE / "outcomes.jsonl"
    if p.exists():
        for l in open(p, encoding="utf8"):
            o = json.loads(l)
            if SECRET.search(o.get("task_text", "")): continue
            out.append(dict(key=o["id"], namespace=NS, tags=["outcome", o.get("agent") or "", o.get("outcome") or ""],
                            value=json.dumps(dict(task=o["task_text"][:400], agent=o.get("agent"), outcome=o.get("outcome"), reward=o.get("reward"), tokens=(o.get("tokens") or {}).get("total")))))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=""); ap.add_argument("--min-value", type=float, default=0.3); a = ap.parse_args()
    b = batch(a.min_value)
    (HERE / "work").mkdir(exist_ok=True)
    with open(HERE / "work" / "ruflo_mirror_batch.jsonl", "w", encoding="utf8") as f:
        for r in b: f.write(json.dumps(r) + "\n")
    print(f"{len(b)} records -> bench/work/ruflo_mirror_batch.jsonl")
    if a.apply:
        ok = 0
        for r in b:
            cmd = ["node", CLI, "memory", "store", "-k", r["key"], "--value", r["value"], "-n", NS, "--tags", ",".join(t for t in r["tags"] if t), "--scan-content"]
            if a.db: cmd += ["--path", a.db]
            ok += subprocess.run(cmd, capture_output=True, text=True, timeout=120).returncode == 0
        print(f"stored {ok}/{len(b)}")
