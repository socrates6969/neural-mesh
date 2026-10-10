"""bench/run.py - run one battery split under up to four conditions and append rows to bench/results/<tag>/raw.jsonl.
  A  claude      : `claude -p` (haiku), no tools, same system prompt.
  B  claude+mesh : same, behind the mesh retrieval front-end (bench/frontend.py; context attached only above its score gate).
  C  local       : Ollama qwen2.5-coder:7b (stand-in for mesh-llm until a promoted adapter exists), temperature 0, seed 0.
  D  local+mesh  : C behind the same front-end.
Units are (split, id, cond, rep): K repeats (default 3) for the Claude conditions A/B, K=1 for the local ones plus a 10-item
determinism probe (rep 1, probe=true, not scored). A/B units of the same (item, rep) run back to back in one worker, order
alternated, so load drift hits both equally.

Every call is classified OK / MODEL / INFRA. INFRA (timeouts, transport and API errors, rate limits, sandbox failures) is
retried up to --max-tries times with exponential backoff and full jitter; a unit still INFRA is written as status infra_fail,
is never scored and is retried by the next (resumed) run. A sandbox outage while scoring a finished answer is INFRA too, but the
row keeps the answer text (sandbox_infra=true) and the resumed run re-scores that text; the model is never called twice. MODEL (truncated or empty completions, refusals) is status
model_fail and scores as wrong. Scoring parses strictly: the last non-empty line must be `ANSWER: <x>` (code: the first fenced
block); the old lenient parse (any ANSWER:, else the last number; whole text as code) is kept as fallback_correct, a
diagnostic only. The full answer text is stored.

results/<tag>/manifest_<split>.json pins git sha + dirty flag, battery/corpus/frontend/PREREG shas, models (+ Ollama digest), K,
sandbox and the relevant env vars; resuming with a different pin exits 2. One writer per tag (O_EXCL lock file, exit 5).
HELD/CONF runs are refused unless the front-end sha, front-end env, battery sha and K match bench/PREREG.json and the sandbox
is docker. A complete run ends by calling analyze.py on the tag.

Usage: python bench/run.py --split tune|held|conf --cond A,B,C,D --tag NAME [--k 3] [--workers 4] [--sandbox docker|job|auto]
                           [--max-tries 5] [--limit N] [--cats math,code] [--dry-run] [--no-analyze]
Exit: 0 ok, 1 usage, 2 manifest/stale/prereg, 3 incomplete (infra_fail left), 4 sandbox unavailable, 5 locked."""
import argparse, json, math, os, pathlib, random, re, subprocess, sys, time, urllib.error, urllib.request, concurrent.futures as cf
HERE = pathlib.Path(__file__).resolve().parent
NM = HERE.parent
sys.path.insert(0, str(HERE))
CLAUDE_MODEL = os.environ.get("BENCH_CLAUDE_MODEL", "haiku")
LOCAL_MODEL = os.environ.get("BENCH_LOCAL_MODEL", "qwen2.5-coder:7b")
OLLAMA = os.environ.get("BENCH_OLLAMA_URL", "http://localhost:11434")
SYSTEM = "Answer concisely and exactly as instructed. Do not use tools."
WORK = HERE / "work" / "cc"
RESULTS = pathlib.Path(os.environ.get("BENCH_RESULTS") or HERE / "results")

from frontend import *  # noqa: F401,F403  (retrieval front-end: build_corpus, BM25, retrieve, retrieve_v2, query_of, hook_frontend, frontend_prompt, THRESH)
from frontend import THRESH, frontend_prompt
import frontend, sandbox
from common import AXES, axis_of, battery_path, file_sha, git_info, json_sha, load_prereg, PREREG_PATH

EXIT_OK, EXIT_USAGE, EXIT_PREREG, EXIT_INCOMPLETE, EXIT_SANDBOX, EXIT_LOCK = 0, 1, 2, 3, 4, 5
CLAUDE_CONDS, LOCAL_CONDS = ("A", "B"), ("C", "D")
ENV_KEYS = ("BENCH_CLAUDE_MODEL", "BENCH_THINK", "BENCH_LOCAL_MODEL", "BENCH_OLLAMA_URL", "MESH_V2", "MESH_THRESH", "MESH_FRONTEND",
            "RUFLO_CLI", "NEURAL_MESH_RUFLO_DB")
PROBE_N = 10


class Abort(Exception):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


# ------------------------------------------------------------------ model calls (each returns a dict with cls = OK | MODEL | INFRA)
INFRA_TEXT = re.compile(r"API Error|rate.?limit|usage limit|overloaded|\b5\d\d\b|timed? ?out|ECONN|ETIMEDOUT|network|socket hang up|"
                        r"connection (?:error|refused|reset)|Internal server error|Service Unavailable", re.I)


def classify_claude(timed_out, stdout, stderr=""):
    """(cls, parsed json or None, err)"""
    if timed_out: return "INFRA", None, "timeout"
    try: j = json.loads(stdout)
    except Exception: return "INFRA", None, ("non-json: " + (stdout + stderr).strip())[:300]
    if not isinstance(j, dict): return "INFRA", None, "non-object json"
    text = j.get("result") or ""
    if j.get("is_error"):
        return ("INFRA" if INFRA_TEXT.search(text + " " + str(j.get("subtype", ""))) or not text else "MODEL"), j, ("is_error: " + text)[:300]
    if j.get("stop_reason") == "end_turn" and text.strip(): return "OK", j, None
    return "MODEL", j, f"stop_reason={j.get('stop_reason')} empty={not text.strip()}"


def call_claude(prompt):
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MAX_THINKING_TOKENS=os.environ.get("BENCH_THINK", "2048"))
    try:
        p = subprocess.run(["claude", "-p", prompt, "--output-format", "json", "--tools", "", "--system-prompt", SYSTEM, "--model", CLAUDE_MODEL,
                            "--no-session-persistence", "--disable-slash-commands", "--setting-sources", "", "--strict-mcp-config"],
                           cwd=str(WORK), env=env, capture_output=True, text=True, encoding="utf8", errors="replace", timeout=240)
        cls, j, err = classify_claude(False, p.stdout, p.stderr)
    except subprocess.TimeoutExpired:
        cls, j, err = classify_claude(True, "")
    except OSError as e:
        cls, j, err = "INFRA", None, "spawn: " + repr(e)[:200]
    wall = time.time() - t0
    if j is None: return dict(cls=cls, text="", wall=wall, err=err, tokens=None)
    u = j.get("usage", {}) or {}
    tok = dict(input=u.get("input_tokens", 0), cache_creation=u.get("cache_creation_input_tokens", 0), cache_read=u.get("cache_read_input_tokens", 0), output=u.get("output_tokens", 0))
    tok["total"] = sum(tok.values())
    return dict(cls=cls, text=j.get("result") or "", wall=wall, api_ms=j.get("duration_api_ms"), ttft_ms=j.get("ttft_ms"), cost_usd=j.get("total_cost_usd"),
                tokens=tok, stop=j.get("stop_reason"), model_ids=sorted((j.get("modelUsage") or {}).keys()), err=err)


def call_ollama(prompt, model=None):
    model = model or LOCAL_MODEL
    t0 = time.time()
    body = json.dumps(dict(model=model, system=SYSTEM, prompt=prompt, stream=False, options=dict(temperature=0, seed=0, num_predict=1200))).encode()
    try:
        r = json.loads(urllib.request.urlopen(urllib.request.Request(f"{OLLAMA}/api/generate", body, {"Content-Type": "application/json"}), timeout=600).read())
    except Exception as e:
        return dict(cls="INFRA", text="", wall=time.time() - t0, err=repr(e)[:200], tokens=None)
    tok = dict(input=r.get("prompt_eval_count", 0), output=r.get("eval_count", 0)); tok["total"] = tok["input"] + tok["output"]
    text = r.get("response", "") or ""
    cls = "OK" if r.get("done_reason") == "stop" and text.strip() else "MODEL"
    return dict(cls=cls, text=text, wall=time.time() - t0, api_ms=(r.get("total_duration") or 0) / 1e6, tokens=tok, stop=r.get("done_reason"),
                model_ids=[model], err=None if cls == "OK" else f"done_reason={r.get('done_reason')}")


def backoff(attempt, base=2.0, cap=60.0, rng=random):
    """full-jitter exponential backoff before retry number `attempt` (1-based)"""
    return rng.uniform(0, min(cap, base * 2 ** (attempt - 1)))


def call_with_retry(fn, max_tries=5, sleep=time.sleep, rng=random, base=2.0, cap=60.0):
    """call fn() until it is not INFRA or max_tries are used; returns (result, attempts, infra_errors)"""
    errs = []
    for k in range(1, max_tries + 1):
        r = fn()
        if r.get("cls") != "INFRA": return r, k, errs
        errs.append(r.get("err"))
        if k < max_tries: sleep(backoff(k, base, cap, rng))
    return r, max_tries, errs


# ------------------------------------------------------------------ parsing + scoring
ANS_LINE = re.compile(r"^[\s>*_`#-]*ANSWER\s*:\s*(.*?)\s*$")
CODE_STRICT = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.S)


def _clean(a):
    a = a.strip().rstrip(".").strip().strip("`*_\"' ").rstrip(".").strip()
    return a


def extract_strict(text):
    """the answer only if the LAST non-empty line is `ANSWER: <x>` (markdown emphasis tolerated); else ''"""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    if not lines: return ""
    m = ANS_LINE.match(lines[-1])
    return _clean(m.group(1)) if m else ""


def extract_answer(text):
    """legacy lenient parse (v1/v2 runs): the last `ANSWER:` anywhere"""
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return (m[-1].strip() if m else "").strip("`*\" '").rstrip(".")


def match_answer(item, ans):
    if not ans: return False
    if item["kind"] == "num":
        try:
            got = float(ans.replace(",", "").replace("_", "")); exp = float(item["answer"])
        except Exception: return False
        tol = item.get("tol", 0) or 0
        return math.isclose(got, exp, rel_tol=max(tol, 1e-9), abs_tol=1e-12)
    return ans.lower().strip("`'\" ") == str(item["answer"]).lower()


def code_of(text, strict=True):
    m = CODE_STRICT.findall(text or "")
    if m: return m[0]
    return None if strict else (text or "")


def score(item, text, mode="job", max_tries=5, sleep=time.sleep):
    """dict(correct, fallback_correct, parsed, answer, sbx, infra). infra=True means scoring itself failed (sandbox down)."""
    if item["kind"] == "code":
        out = dict(answer=None, sbx=None, infra=False)
        strict, loose = code_of(text, True), code_of(text, False)
        res = {}
        for key, code in (("strict", strict), ("loose", loose)):
            if code is None: res[key] = None; continue
            if key == "loose" and code == strict: res[key] = res["strict"]; continue
            r = None
            for k in range(1, max_tries + 1):         # retry the sandbox only (the model answer is already paid for)
                r = sandbox.run_code(code, item["fn"], item["tests"], item["expected"], mode=mode)
                if r["status"] != "infra": break
                if k < max_tries: sleep(backoff(k))
            res[key] = r
        if any(r is not None and r["status"] == "infra" for r in res.values()):
            out.update(correct=False, fallback_correct=False, parsed=strict is not None, infra=True, sbx=res.get("strict") or res.get("loose"))
            return out
        out.update(correct=bool(res["strict"] and res["strict"]["status"] == "pass"), fallback_correct=bool(res["loose"] and res["loose"]["status"] == "pass"),
                   parsed=strict is not None, sbx={k: (v and {kk: v.get(kk) for kk in ("status", "bad", "n", "why", "secs")}) for k, v in res.items()})
        return out
    ans = extract_strict(text)
    loose = extract_answer(text)
    if not loose:
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text or "")
        loose = nums[-1] if (nums and item["kind"] == "num") else ""
    return dict(correct=match_answer(item, ans), fallback_correct=match_answer(item, loose), parsed=bool(ans), answer=ans, sbx=None, infra=False)


# ------------------------------------------------------------------ one unit
def run_unit(item, cond, rep, split, sbx_mode, max_tries=5, probe=False, sleep=time.sleep, callers=None):
    callers = callers or {"claude": call_claude, "ollama": call_ollama}
    prompt = item["prompt"]; meta = {}
    if cond in ("B", "D"): prompt, meta = frontend_prompt(prompt)
    fn = (lambda: callers["ollama"](prompt)) if cond in LOCAL_CONDS else (lambda: callers["claude"](prompt))
    r, attempts, infra_errs = call_with_retry(fn, max_tries=max_tries, sleep=sleep)
    row = dict(split=split, id=item["id"], cat=item["cat"], sub=item["sub"], axis=axis_of(item), cond=cond, rep=rep, probe=probe, attempts=attempts,
               cls=r["cls"], wall_s=r.get("wall"), api_ms=r.get("api_ms"), tokens=r.get("tokens"), cost_usd=r.get("cost_usd"), stop=r.get("stop"),
               model_ids=r.get("model_ids"), err=r.get("err"), infra_errs=[e for e in infra_errs if e][:5], text=r.get("text", ""), ts=time.time(), **meta)
    if r["cls"] == "INFRA":
        row.update(status="infra_fail", correct=None, fallback_correct=None, parsed=None, completed=False)
        return row
    return finish_row(row, item, r["cls"], sbx_mode, max_tries, sleep)


def finish_row(row, item, model_cls, sbx_mode, max_tries=5, sleep=time.sleep):
    """score the stored answer text of a finished model call. A sandbox outage leaves the row infra_fail with sandbox_infra=True
    and the text kept, so a resumed run re-scores that text (rescore_row) and never calls the model again."""
    sc = score(item, row["text"], mode=sbx_mode, max_tries=max_tries, sleep=sleep)
    if sc["infra"]:
        row.setdefault("model_err", row.get("err"))
        row.update(status="infra_fail", cls="INFRA", model_cls=model_cls, sandbox_infra=True, err="sandbox: " + str((sc.get("sbx") or {}).get("why")),
                   correct=None, fallback_correct=None, parsed=sc["parsed"], completed=True)
        return row
    row.update(cls=model_cls, sandbox_infra=False)
    if model_cls == "MODEL":
        row.update(status="model_fail", correct=False, fallback_correct=sc["fallback_correct"], parsed=sc["parsed"], answer=sc.get("answer"), sbx=sc.get("sbx"), completed=False)
    else:
        row.update(status="ok", err=None, correct=sc["correct"], fallback_correct=sc["fallback_correct"], parsed=sc["parsed"], answer=sc.get("answer"), sbx=sc.get("sbx"), completed=True)
    return row


def rescore_row(old, item, sbx_mode, max_tries=5, sleep=time.sleep):
    """a new row for a sandbox-infra unit: same answer text and model metadata, scored again; no model call"""
    row = {k: v for k, v in old.items() if k not in ("status", "correct", "fallback_correct", "parsed", "answer", "sbx", "completed", "_tag")}
    row.update(ts=time.time(), rescored=int(old.get("rescored", 0)) + 1)
    if row.get("model_cls") == "MODEL": row["err"] = old.get("model_err")
    return finish_row(row, item, old.get("model_cls") or "OK", sbx_mode, max_tries, sleep)


def pending_rescore(path, split):
    """{(id, cond, rep): last row} for units whose model answer is stored but whose sandbox scoring hit INFRA"""
    last, final = {}, set()
    if path.exists():
        for l in open(path, encoding="utf8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("split", "tune") != split: continue
            key = (r["id"], r["cond"], r.get("rep", 0))
            if r.get("status") in ("ok", "model_fail"): final.add(key)
            last[key] = r
    return {k: r for k, r in last.items() if k not in final and r.get("status") == "infra_fail" and r.get("sandbox_infra") and r.get("text")}


# ------------------------------------------------------------------ planning, manifest, lock
def plan_units(items, conds, k, done, probe_n=PROBE_N):
    """-> list of work packets: ('pair', item, rep, order) for A+B, ('one', item, cond, rep, probe)"""
    packets = []
    ordered = sorted(items, key=lambda i: i["id"])
    probe_ids = {i["id"] for i in ordered[:probe_n]}
    claude = [c for c in CLAUDE_CONDS if c in conds]
    for ix, it in enumerate(ordered):
        for rep in range(k):
            need = [c for c in claude if (it["id"], c, rep) not in done]
            if len(need) == 2:
                packets.append(("pair", it, rep, ["A", "B"] if (ix + rep) % 2 == 0 else ["B", "A"]))
            for c in (need if len(need) == 1 else []):
                packets.append(("one", it, c, rep, False))
    for c in [c for c in LOCAL_CONDS if c in conds]:
        for it in ordered:
            if (it["id"], c, 0) not in done: packets.append(("one", it, c, 0, False))
        for it in ordered:
            if it["id"] in probe_ids and (it["id"], c, 1) not in done: packets.append(("one", it, c, 1, True))
    return packets


def done_units(path, split):
    """units with a final status (ok | model_fail); infra_fail rows are retried"""
    done = set()
    if path.exists():
        for l in open(path, encoding="utf8"):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get("split", "tune") == split and r.get("status") in ("ok", "model_fail"):
                done.add((r["id"], r["cond"], r.get("rep", 0)))
    return done


def ollama_digest(model=None):
    model = model or LOCAL_MODEL
    try:
        tags = json.loads(urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10).read())
        for m in tags.get("models", []):
            if m.get("name") == model or m.get("model") == model: return m.get("digest")
    except Exception:
        return None
    return None


def claude_cli_version():
    try: return subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception: return None


MANIFEST_PINNED = ("split", "battery_sha", "corpus_sha", "frontend_sha", "frontend_env", "prereg_sha", "conds", "models", "K", "sandbox", "image", "env")


def build_manifest(split, header, conds, k, sbx_mode, prereg_sha):
    models = {}
    if any(c in conds for c in CLAUDE_CONDS): models["claude"] = CLAUDE_MODEL
    if any(c in conds for c in LOCAL_CONDS): models.update(local=LOCAL_MODEL, local_digest=ollama_digest())
    m = dict(split=split, battery_sha=header["items_sha"], corpus_sha=frontend.corpus_sha(), frontend_sha=frontend.frontend_sha(),
             frontend_env=frontend.frontend_env(), prereg_sha=prereg_sha, conds=sorted(conds), models=models, K=dict(claude=k, local=1, probe=PROBE_N),
             sandbox=sbx_mode, image=sandbox.IMAGE if sbx_mode == "docker" else None, env={kk: os.environ.get(kk, "") for kk in ENV_KEYS},
             python=sys.version.split()[0], platform=sys.platform, claude_cli=claude_cli_version() if "claude" in models else None, **git_info())
    return m


def check_manifest(path, new):
    """create on first run; on resume every pinned field must match (exit 2)"""
    if not path.exists():
        path.write_text(json.dumps(dict(new, created=time.time()), indent=1), encoding="utf8"); return
    old = json.load(open(path, encoding="utf8"))
    diff = [k for k in MANIFEST_PINNED if old.get(k) != new.get(k)]
    if diff:
        raise Abort(EXIT_PREREG, "manifest mismatch on resume (" + ", ".join(f"{k}: {old.get(k)!r} != {new.get(k)!r}" for k in diff) + ")")


class Lock:
    def __init__(self, path): self.path = path; self.fd = None
    def __enter__(self):
        try: self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError: raise Abort(EXIT_LOCK, f"locked: {self.path} exists (another writer, or a crashed run: delete it if no run is active)")
        os.write(self.fd, json.dumps(dict(pid=os.getpid(), t=time.time())).encode()); return self
    def __exit__(self, *a):
        os.close(self.fd)
        try: os.unlink(self.path)
        except OSError: pass


def prereg_checks(split, header, conds, k, sbx_mode):
    """HELD/CONF only run exactly what PREREG.json froze. Returns prereg_sha."""
    pre, pre_sha = load_prereg()
    if split == "tune": return pre_sha
    if pre is None: raise Abort(EXIT_PREREG, f"{split} refused: {PREREG_PATH.name} missing")
    if frontend.frontend_sha() != pre["frontend_sha"]: raise Abort(EXIT_PREREG, f"{split} refused: frontend_sha {frontend.frontend_sha()[:12]} != PREREG {pre['frontend_sha'][:12]}")
    if frontend.frontend_env() != pre["frontend_env"]: raise Abort(EXIT_PREREG, f"{split} refused: front-end env {frontend.frontend_env()} != PREREG {pre['frontend_env']}")
    if header["items_sha"] != pre["battery"][split]["items_sha"]: raise Abort(EXIT_PREREG, f"{split} refused: battery_{split}.json items_sha differs from PREREG")
    if k != pre["K"]["claude"]: raise Abort(EXIT_PREREG, f"{split} refused: K={k} but PREREG K={pre['K']['claude']}")
    if not set(conds) <= set(pre["conds"]): raise Abort(EXIT_PREREG, f"{split} refused: conds {conds} not in PREREG {pre['conds']}")
    if sbx_mode != "docker": raise Abort(EXIT_PREREG, f"{split} refused: sandbox must be docker (job mode is not a security boundary)")
    return pre_sha


def stale_check(split, header, items):
    import battery
    problems = battery.check_items(split, header, items)
    stale = [p for p in problems if p[0] == "stale"]
    for p in problems:
        if p[0] == "warn": print("warning:", p[1], file=sys.stderr)
    if stale: raise Abort(EXIT_PREREG, f"battery_{split}.json is stale: " + "; ".join(p[1] for p in stale[:5]))


# ------------------------------------------------------------------ main
class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr); print(f"{self.prog}: error: {message}", file=sys.stderr); sys.exit(EXIT_USAGE)


def parse(argv):
    ap = Parser(prog="run.py")
    ap.add_argument("--split", required=True, choices=["tune", "held", "conf"]); ap.add_argument("--cond", default="A,B")
    ap.add_argument("--tag", required=True); ap.add_argument("--k", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sandbox", default="auto", choices=["auto", "docker", "job"]); ap.add_argument("--max-tries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--cats", default=""); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    a = ap.parse_args(argv)
    conds = [c for c in a.cond.split(",") if c]
    if not conds or any(c not in "ABCD" or len(c) != 1 for c in conds): ap.error(f"--cond must be a subset of A,B,C,D: {a.cond}")
    if a.k < 1 or a.max_tries < 1 or a.workers < 1: ap.error("--k, --max-tries and --workers must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.tag): ap.error("--tag: letters, digits, _ . - only")
    if a.split != "tune" and (a.limit or a.cats): ap.error("--limit/--cats are TUNE-only (HELD/CONF run the whole pre-registered battery)")
    a.conds = conds
    return a


def select(items, a):
    if a.cats: items = [i for i in items if i["cat"] in a.cats.split(",") or axis_of(i) in a.cats.split(",")]
    if a.limit:  # stratified: first N per axis
        seen, sel = {}, []
        for i in items:
            seen[axis_of(i)] = seen.get(axis_of(i), 0) + 1
            if seen[axis_of(i)] <= a.limit: sel.append(i)
        items = sel
    return items


def resolve_sandbox(mode, split):
    if mode == "auto": mode = "docker" if split != "tune" or sandbox.docker_available()[0] else "job"
    return mode


def execute(a, callers=None, sleep=time.sleep):
    bp = battery_path(a.split)
    if not bp.exists(): raise Abort(EXIT_PREREG, f"{bp.name} missing: run python bench/battery.py --split {a.split}")
    j = json.load(open(bp, encoding="utf8")); header, items = j["header"], j["items"]
    if json_sha(items) != header["items_sha"]: raise Abort(EXIT_PREREG, f"{bp.name}: items_sha does not match its contents")
    stale_check(a.split, header, items)
    sbx_mode = resolve_sandbox(a.sandbox, a.split)
    prereg_sha = prereg_checks(a.split, header, a.conds, a.k, sbx_mode)
    items = select(items, a)
    out = RESULTS / a.tag; raw = out / "raw.jsonl"
    done = done_units(raw, a.split)
    by_id = {i["id"]: i for i in items}
    rescore = {k: r for k, r in pending_rescore(raw, a.split).items() if k[0] in by_id and k[1] in a.conds}
    packets = plan_units(items, a.conds, a.k, done | set(rescore))
    n_calls = sum(2 if p[0] == "pair" else 1 for p in packets)
    n_claude = sum(2 if p[0] == "pair" else int(p[2] in CLAUDE_CONDS) for p in packets)
    print(f"plan: split={a.split} tag={a.tag} items={len(items)} conds={','.join(a.conds)} K={a.k} sandbox={sbx_mode} already-done={len(done)} sandbox-rescore={len(rescore)} "
          f"units-to-run={n_calls} (claude calls {n_claude}, est. ${n_claude * 0.0031:.2f} at the v2 mean of $0.0031/call before retries)", flush=True)
    if a.dry_run: return EXIT_OK
    if sbx_mode == "docker":
        ok, why = sandbox.docker_available()
        if not ok: raise Abort(EXIT_SANDBOX, f"docker sandbox unavailable: {why}")
    out.mkdir(parents=True, exist_ok=True)
    with Lock(out / ".lock"):
        check_manifest(out / f"manifest_{a.split}.json", build_manifest(a.split, header, a.conds, a.k, sbx_mode, prereg_sha))
        mt = a.max_tries

        def work(p):
            if p[0] == "pair":
                _, it, rep, order = p
                return [run_unit(it, c, rep, a.split, sbx_mode, mt, sleep=sleep, callers=callers) for c in order]
            _, it, c, rep, probe = p
            return [run_unit(it, c, rep, a.split, sbx_mode, mt, probe=probe, sleep=sleep, callers=callers)]
        t0 = time.time(); n_inf = 0
        claude_p = [p for p in packets if p[0] == "pair" or p[2] in CLAUDE_CONDS]
        local_p = [p for p in packets if p not in claude_p]
        with open(raw, "a", encoding="utf8") as fh:       # the main thread is the only writer
            for (iid, c, rep), old in sorted(rescore.items()):   # sandbox INFRA earlier: re-score the stored answer, no model call
                r = rescore_row(old, by_id[iid], sbx_mode, mt, sleep=sleep)
                n_inf += r["status"] == "infra_fail"
                fh.write(json.dumps(r) + "\n"); fh.flush()
            for group, w in ((claude_p, a.workers), (local_p, 1)):   # GPU: one local call at a time so latency is not contended
                if not group: continue
                with cf.ThreadPoolExecutor(w) as ex:
                    for fut in cf.as_completed([ex.submit(work, p) for p in group]):
                        for r in fut.result():
                            n_inf += r["status"] == "infra_fail"
                            fh.write(json.dumps(r) + "\n")
                        fh.flush()
        print(f"ran {n_calls} units in {time.time() - t0:.0f}s; infra_fail left: {n_inf}", flush=True)
    if n_inf:
        print(f"incomplete: {n_inf} units still infra_fail; re-run the same command to retry them", file=sys.stderr)
        return EXIT_INCOMPLETE
    if a.no_analyze: return EXIT_OK
    import analyze
    return analyze.main([a.tag])


def main(argv=None, callers=None, sleep=time.sleep):
    a = parse(sys.argv[1:] if argv is None else argv)
    try:
        return execute(a, callers=callers, sleep=sleep)
    except Abort as e:
        print(f"run.py: {e}", file=sys.stderr); return e.code


if __name__ == "__main__":
    sys.exit(main())
