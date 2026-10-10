"""bench/run.py - run the battery under three conditions and write bench/results/<tag>/raw.jsonl.
  A  claude      : `claude -p` (haiku-4-5), no tools, same system prompt.
  B  claude+mesh : same, but the mesh front-end first retrieves repo context (BM25 over neural-mesh docs+src);
                   context is attached only when the top BM25 score >= THRESH (otherwise the prompt is unchanged).
  D  local+mesh  : local model behind the same retrieval front-end (shows what retrieval adds to a weak model).
  C  local       : Ollama qwen2.5-coder:7b (stand-in for mesh-llm until a promoted adapter exists), temperature 0.
Tokens come from the API usage fields in the claude JSON (input + cache_creation + cache_read + output);
latency is wall-clock around the subprocess plus the API-reported duration_ms. Ollama tokens come from
prompt_eval_count / eval_count, latency from wall-clock.
Usage: python bench/run.py --cond A,B,C [--limit N] [--workers 4] [--tag name]"""
import argparse, json, math, os, pathlib, re, subprocess, sys, time, urllib.request, concurrent.futures as cf
HERE = pathlib.Path(__file__).resolve().parent
NM = HERE.parent
CLAUDE_MODEL = os.environ.get("BENCH_CLAUDE_MODEL", "haiku")
SYSTEM = "Answer concisely and exactly as instructed. Do not use tools."
WORK = HERE / "work" / "cc"; WORK.mkdir(parents=True, exist_ok=True)

from frontend import *  # noqa: F401,F403  (retrieval front-end: build_corpus, BM25, retrieve, retrieve_v2, query_of, hook_frontend, frontend_prompt, THRESH)
from frontend import THRESH, frontend_prompt


# ------------------------------------------------------------------ model calls
def call_claude(prompt):
    t0 = time.time()
    env = dict(os.environ, MAX_THINKING_TOKENS=os.environ.get("BENCH_THINK", "2048"))
    try:
      p = subprocess.run(["claude", "-p", prompt, "--output-format", "json", "--tools", "", "--system-prompt", SYSTEM, "--model", CLAUDE_MODEL,
                        "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "", "--strict-mcp-config"],
                       cwd=str(WORK), env=env, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=240)
    except subprocess.TimeoutExpired:
      return dict(text="", ok=False, wall=time.time() - t0, err="timeout240", tokens=None)
    wall = time.time() - t0
    try: j = json.loads(p.stdout)
    except Exception: return dict(text="", ok=False, wall=wall, err=(p.stdout + p.stderr)[:300], tokens=None)
    u = j.get("usage", {})
    tok = dict(input=u.get("input_tokens", 0), cache_creation=u.get("cache_creation_input_tokens", 0), cache_read=u.get("cache_read_input_tokens", 0), output=u.get("output_tokens", 0))
    tok["total"] = sum(tok.values())
    return dict(text=j.get("result") or "", ok=not j.get("is_error", False) and j.get("stop_reason") == "end_turn", wall=wall,
                api_ms=j.get("duration_api_ms"), ttft_ms=j.get("ttft_ms"), cost_usd=j.get("total_cost_usd"), tokens=tok, stop=j.get("stop_reason"), err=None)

def call_ollama(prompt, model=os.environ.get("BENCH_LOCAL_MODEL", "qwen2.5-coder:7b")):
    t0 = time.time()
    body = json.dumps(dict(model=model, system=SYSTEM, prompt=prompt, stream=False, options=dict(temperature=0, seed=0, num_predict=1200))).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request("http://localhost:11434/api/generate", body, {"Content-Type": "application/json"}), timeout=600).read())
    except Exception as e:
        return dict(text="", ok=False, wall=time.time() - t0, err=str(e)[:200], tokens=None)
    tok = dict(input=r.get("prompt_eval_count", 0), output=r.get("eval_count", 0)); tok["total"] = tok["input"] + tok["output"]
    return dict(text=r.get("response", ""), ok=r.get("done_reason") == "stop", wall=time.time() - t0, api_ms=(r.get("total_duration") or 0) / 1e6, tokens=tok, err=None)

# ------------------------------------------------------------------ scoring
CODE_HARNESS = r'''
import json, sys
cand, ref, fn, tests = json.load(sys.stdin)
g1, g2 = {}, {}
exec(ref, g2)
try:
    exec(cand, g1)
except Exception as e:
    print(json.dumps({"ok": False, "why": "compile:" + repr(e)[:80]})); sys.exit()
bad = 0
for t in tests:
    try: exp = g2[fn](*json.loads(json.dumps(t)))
    except Exception as e: exp = "EXC"
    try: got = g1[fn](*json.loads(json.dumps(t)))
    except Exception as e: got = "EXC:" + repr(e)[:40]
    if json.dumps(got, sort_keys=True, default=str) != json.dumps(exp, sort_keys=True, default=str): bad += 1
print(json.dumps({"ok": bad == 0, "bad": bad, "n": len(tests)}))
'''
def extract_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    return (m[-1].strip() if m else "").strip("`*\" '").rstrip(".")
def score(item, text):
    if item["kind"] == "code":
        m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
        code = m[0] if m else text
        try:
            p = subprocess.run([sys.executable, "-c", CODE_HARNESS], input=json.dumps([code, item["ref"], item["fn"], item["tests"]]), capture_output=True, text=True, timeout=20)
            return bool(json.loads(p.stdout.strip().splitlines()[-1])["ok"])
        except Exception: return False
    ans = extract_answer(text)
    if not ans:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
        ans = nums[-1] if (nums and item["kind"] == "num") else ""
    if item["kind"] == "num":
        try:
            got = float(ans.replace(",", "").replace("_", "")); exp = float(item["answer"])
        except Exception: return False
        tol = item.get("tol", 0) or 0
        return math.isclose(got, exp, rel_tol=max(tol, 1e-9), abs_tol=1e-12)
    return ans.lower().strip("`'\" ") == str(item["answer"]).lower()

# ------------------------------------------------------------------ main
def run_one(item, cond):
    prompt = item["prompt"]; meta = {}
    if cond in ("B", "D"): prompt, meta = frontend_prompt(prompt)
    r = call_ollama(prompt) if cond in ("C", "D") else call_claude(prompt)
    ok_fmt = bool(r["text"].strip()) and r["ok"]
    correct = score(item, r["text"]) if ok_fmt else False
    parsed = bool(re.findall(r"```(?:python)?[\r\n]+(.*?)```", r["text"], re.S)) if item["kind"] == "code" else bool(extract_answer(r["text"]))
    return dict(id=item["id"], cat=item["cat"], sub=item["sub"], cond=cond, correct=bool(correct), completed=bool(ok_fmt), parsed=parsed, wall_s=r["wall"], api_ms=r.get("api_ms"),
                tokens=r["tokens"], cost_usd=r.get("cost_usd"), err=r.get("err"), text=r["text"][:1500], **meta)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cond", default="A,B,C"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--tag", default="main"); ap.add_argument("--cats", default="")
    a = ap.parse_args()
    items = json.load(open(HERE / "battery.json"))
    if a.cats: items = [i for i in items if i["cat"] in a.cats.split(",")]
    if a.limit:  # stratified: first N per category
        seen = {}; sel = []
        for i in items:
            seen[i["cat"]] = seen.get(i["cat"], 0) + 1
            if seen[i["cat"]] <= a.limit: sel.append(i)
        items = sel
    out = HERE / "results" / a.tag; out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "raw.jsonl").exists():
        for l in open(out / "raw.jsonl", encoding="utf8"):
            r = json.loads(l); done.add((r["id"], r["cond"]))
    conds = a.cond.split(",")
    def pair(ix_item):
        ix, it = ix_item   # A/B back-to-back in the same worker, order alternated, so load drift hits both equally
        order = ["A", "B"] if ix % 2 == 0 else ["B", "A"]
        return [run_one(it, c) for c in order]
    if "A" in conds and "B" in conds:
        todo = [it for it in items if (it["id"], "A") not in done or (it["id"], "B") not in done]
        t0 = time.time()
        with cf.ThreadPoolExecutor(a.workers) as ex, open(out / "raw.jsonl", "a", encoding="utf8") as fh:
            for rs in ex.map(pair, list(enumerate(todo))):
                for r in rs: fh.write(json.dumps(r) + "\n")
                fh.flush()
        print(f"pairs A+B: {len(todo)} items in {time.time()-t0:.0f}s", flush=True)
        conds = [c for c in conds if c not in ("A", "B")]
    for cond in conds:
        todo = [i for i in items if (i["id"], cond) not in done]
        w = 1 if cond in ("C", "D") else a.workers   # GPU: one at a time so latency is not contended
        t0 = time.time()
        with cf.ThreadPoolExecutor(w) as ex, open(out / "raw.jsonl", "a", encoding="utf8") as fh:
            for r in ex.map(lambda it: run_one(it, cond), todo):
                fh.write(json.dumps(r) + "\n"); fh.flush()
        print(f"cond {cond}: {len(todo)} items in {time.time()-t0:.0f}s", flush=True)
