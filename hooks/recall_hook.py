#!/usr/bin/env python
"""Claude Code UserPromptSubmit hook: inject a few relevant memories from the lifelong MemoryStore (opt-in, silent on any problem).

stdin : the hook JSON ({"prompt", "cwd", "session_id", ...}).   stdout: a short block that Claude Code adds to the model context,
or NOTHING.  Always exits 0.  Disable with NEURAL_MESH_RECALL=off (checked before any import: ~35 ms for `python -S`).

Pipeline (hooks/README.md has the numbers):
  1. cheap gate  : is the prompt about the user's code / projects?  (paths, backticked or code-like identifiers, a known project name,
                   "last time / we decided / our convention" phrases).  Chit-chat, slash commands and 1-3 word prompts stop here.
  2. pool        : memories of the project the session runs in (cwd basename) plus projects named in the prompt.
  3. recall      : ingest.memory.MemoryStore.recall (HashEmbedding cosine x value), k<=3, then a RELEVANCE FLOOR: cosine >= MIN_COS and
                   a lexical check (>= 2 shared rare tokens, or 1 shared identifier-like token).  Nothing passes -> print nothing.
  4. only the memories that are actually printed are logged (session id) so the nightly ingest can credit them by outcome.
  5. every printed byte goes through ingest.scrub again (a SECRET-class hit drops the memory; the whole block is re-checked).
Hard limits: internal budget BUDGET_S (default 1.2 s, env NEURAL_MESH_RECALL_BUDGET_S) enforced by a watchdog thread + os._exit, so the
Claude Code hook timeout (3 s in hooks/settings-hooks-recall.json) always exceeds it; output <= MAX_CHARS (~600 tokens worst case).
"""
import os
import sys

if os.environ.get("NEURAL_MESH_RECALL", "").strip().lower() == "off":
    sys.exit(0)

import json
import re
import threading
import time

T0 = time.monotonic()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_S = 1.2
MAX_K = 3
MIN_COS = 0.30           # recall()'s own cosine floor; the lexical score below is the real discriminator (hooks/README.md)
MIN_STRONG = 3           # shared tokens that are RARE in the pool (df <= max(2, RARE_DF * pool size)) needed ...
RARE_DF = 0.05
IDENT_DF = 0.10          # ... or one shared identifier-like token with df <= max(2, IDENT_DF * pool size)
QUERY_CHARS = 600        # only the head of a long (pasted) prompt is matched
MAX_CHARS = 1500         # <= ~600 tokens even at 2.5 chars/token
ITEM_CHARS = 420
MIN_GATE_SCORE = 2

_EXT = r"(?:py|rs|ts|tsx|js|jsx|mjs|cjs|json|md|toml|ya?ml|sql|ps1|sh|go|java|css|html|lock|cfg|ini|env)"
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\./|\.\./|~/|\b[\w.-]+/[\w.-]+/|\bsrc/|\b[\w-]+\." + _EXT + r"\b)")
_BACKTICK = re.compile(r"`(?=[^`\s]{2,80}`)[^`\s]+`")   # a code-like span: no whitespace (so `ANSWER: <number>` does not count)
_IDENT = re.compile(r"\b(?:[a-z][a-z0-9]*_[a-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]+|\w+\.\w+\(|\w+::\w+)|--[a-z][a-z-]+")
_PHRASE = re.compile(r"(?i)\b(?:last time|we decided|we agreed|our convention|the convention|as before|earlier we|previously|remember (?:that|how|when|the)|"
                     r"what did we|where did we|why did we|how did we|did we (?:decide|already|ever))\b")
_WEAK = re.compile(r"(?i)\b(?:repo|repository|codebase|branch|commit|migration|schema|endpoint|test suite|tests?|pipeline|config(?:uration)?|"
                   r"function|module|class|bug|refactor|deploy|build|hook|router|plugin|agent|memory|state|worktree|ci)\b")
_ACK = re.compile(r"(?i)^\W*(?:continue|yes|no|ok(?:ay)?|thanks?|thank you|go on|go ahead|proceed|sure|please|do it|done|stop|y|n)\W*$")
_STOP = set(("the a an and or but if then else for of in on at to from by with without about into over under is are was were be been being do does did "
             "have has had it its this that these those i me my we our you your they them their he she not no can could should would will shall may might "
             "what when where which who why how all any some more most other such only own same so than too very just also here there now new get got "
             "make made use used using want need like please let know see look find check add change update fix run created called built wrote added "
             "removed fixed changed started finished working worked works").split())
_TOK = re.compile(r"[a-z0-9_]+")


def tokens(text):
    return [t for t in _TOK.findall(text.lower()) if len(t) >= 3 and t not in _STOP and not t.isdigit()]


def _identlike(t):
    return "_" in t or (any(c.isdigit() for c in t) and any(c.isalpha() for c in t))


def gate_score(prompt, projects=()):
    """Cheap relevance-to-the-user's-work score. >= MIN_GATE_SCORE passes. Pure function."""
    p = (prompt or "").strip()
    if len(p) < 12 or p.startswith("/") or _ACK.match(p) or len(p.split()) < 3:
        return 0
    s = 2 * len(_PATH.findall(p)) + 2 * len(_BACKTICK.findall(p)) + 2 * len(_IDENT.findall(p)) + 2 * len(_PHRASE.findall(p))
    low = p.lower()
    s += 2 * sum(1 for pr in projects if _named(pr, low))
    s += min(2, len(_WEAK.findall(p)))
    return s


def project_of_cwd(cwd):
    parts = [x for x in re.split(r"[\\/]", (cwd or "").rstrip("\\/")) if x]
    return parts[-1] if parts else ""


def _named(project, low):
    """A project counts as 'named in the prompt' only for distinctive names (hyphenated/underscored or >= 7 chars): not 'repo', 'mariu'."""
    if not project or not (len(project) >= 7 or (len(project) >= 5 and re.search(r"[-_]", project))):
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(project.lower()) + r"(?![a-z0-9])", low) is not None


def select_pool(items, cwd, prompt):
    """Active memories of the session's own project (cwd basename) plus projects whose name appears in the prompt."""
    low = (prompt or "").lower()
    keep = {project_of_cwd(cwd).lower()} | {i["project"].lower() for i in items if _named(i.get("project"), low)}
    return [i for i in items if i.get("status") == "active" and (i.get("project") or "").lower() in keep]


def lexical_ok(prompt_toks, item_text, df, n):
    """>= MIN_STRONG shared tokens that are rare in the pool, or one shared identifier-like token that is not common (so generic words
    that happen to occur in both texts do not count)."""
    shared = set(prompt_toks) & set(tokens(item_text))
    rare = [t for t in shared if df.get(t, 0) <= max(2, RARE_DF * n)]
    return len(rare) >= MIN_STRONG or any(_identlike(t) and df.get(t, 0) <= max(2, IDENT_DF * n) for t in shared)


def format_block(picked):
    """Scrub again, cap the size. Returns (block, printed_items); ('', []) when nothing is safe to print."""
    from ingest.scrub import scrub
    head = "Notes recalled from earlier sessions (auto-retrieved by neural-mesh; may be outdated, verify before relying on them):"
    lines, used = [], len(head)
    for it in picked:
        r = scrub(it["text"])
        if r.secret:
            continue
        txt = re.sub(r"\s+", " ", r.text).strip()
        if len(txt) > ITEM_CHARS:
            txt = txt[:ITEM_CHARS].rsplit(" ", 1)[0] + " ..."
        line = f"- [{it['tag']}] {txt}"
        if used + len(line) + 1 > MAX_CHARS:
            break
        lines.append((it, line))
        used += len(line) + 1
    if not lines:
        return "", []
    block = head + "\n" + "\n".join(l for _, l in lines)
    if scrub(block).secret:
        return "", []
    return block, [it for it, _ in lines]


def build_block(prompt, cwd, session, cfg=None, log=True):
    """Everything except stdin/stdout/timeouts. Returns the text to print ('' = print nothing)."""
    from ingest.config import Config
    from ingest.memory import MemoryStore
    cfg = cfg or Config()
    if not (cfg.memory_dir / "items.jsonl").exists():
        return ""
    store = MemoryStore(cfg)
    items = list(store.items.values())
    projects = {i.get("project") for i in items if i.get("project")} | {project_of_cwd(cwd)}
    if gate_score(prompt, projects) < MIN_GATE_SCORE:
        return ""
    pool = select_pool(items, cwd, prompt)
    if not pool:
        return ""
    store.items = {i["id"]: i for i in pool}
    query = prompt[:QUERY_CHARS]
    hits = store.recall(query, k=MAX_K * 3, session=session, log=False)
    df = {}
    for i in pool:
        for t in set(tokens(i["text"])):
            df[t] = df.get(t, 0) + 1
    ptoks = [t for t in tokens(query) if t not in {x for pr in projects for x in re.split(r"[-_ ]+", (pr or "").lower())}]
    keep = [h for h in hits if h["cos"] >= MIN_COS and lexical_ok(ptoks, h["text"], df, len(pool))][:MAX_K]
    block, printed = format_block([store.items[h["id"]] for h in keep])
    if block and log:
        store.log_retrieval([i["id"] for i in printed], prompt, session)
    return block


def run_with_deadline(fn, budget_s):
    """Run fn() in a daemon thread; (True, result) or (False, '') after budget_s. Never raises."""
    box = {}

    def work():
        try:
            box["r"] = fn()
        except BaseException:
            box["r"] = ""
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(max(0.0, budget_s))
    return (not th.is_alive()), box.get("r", "")


def main():
    try:
        budget = float(os.environ.get("NEURAL_MESH_RECALL_BUDGET_S", BUDGET_S))
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        prompt, cwd, session = str(data.get("prompt") or ""), str(data.get("cwd") or ""), data.get("session_id")
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        done, out = run_with_deadline(lambda: build_block(prompt, cwd, session), budget - (time.monotonic() - T0))
        if done and out:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stdout.write(out + "\n")
            sys.stdout.flush()
    except BaseException:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
