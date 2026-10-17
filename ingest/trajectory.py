"""Scrubbed per-call tool-use records (gap B), execution evidence (gap C) and per-call taint (gap D).

A call record never holds raw content. It keeps the tool name, the argument SHAPE (key, value type, log10 length bucket), shell
command VERBS (first word plus up to two sub-command words of each command segment, taken outside quoted strings and restricted to
[a-z0-9:._+-]), the file EXTENSION, the scrubbed assistant text that preceded the call, and from the result only ints and booleans:
ok/err, exit code, test passed/failed counts and a failure-marker flag. Arguments and results themselves are never stored.

Execution evidence: a turn (or a subagent) is ``verified`` when a test/build command ran after its last code edit and the last such
command passed (no error flag, no failed tests, no compiler / test-runner failure line in its output); it is ``verified=False`` when
that last check failed, and ``None`` (no evidence) when no check ran after the last edit. A later PASS/FAIL verdict in the same
session can fill a None (pipeline.apply_verdicts). Edits of documentation files (.md, .txt, ...) are not code edits.
"""
from __future__ import annotations

import math
import re

from .scrub import SENSITIVE_PATH_RE, scrub

EDIT_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
READ_TOOLS = {"Read", "NotebookRead"}
SHELL_TOOLS = {"Bash", "bash", "PowerShell", "powershell"}
SEARCH_TOOLS = {"Grep", "Glob", "LS", "ToolSearch"}
WEB_TOOLS = {"WebFetch", "WebSearch"}
DISPATCH_TOOLS = {"Agent", "Task", "SendMessage"}
DOC_EXT = {".md", ".txt", ".rst", ".csv", ".log", ".html", ".htm", ".adoc", ".org"}
MAX_WHY = 800
MAX_VERBS = 6

_NAME_OK = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")
_VERB_OK = re.compile(r"^[a-z][a-z0-9:._+-]{0,24}$")
_SUB_OK = re.compile(r"^(?:[a-z][a-z0-9_-]{0,24}|--?test|--build|-m)$")
_RUN_ARG_OK = re.compile(r"^[a-z][a-z0-9:_.-]{0,40}$")       # npm run <script> / python -m <module>
_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SKIP_VERBS = {"cd", "export", "set", "source", ".", "echo", "pushd", "popd", "sleep", "true", "false", "env", "time", "timeout", "nice",
               "then", "do", "done", "fi", "else", "elif", "if", "for", "while", "until", "case", "esac", "{", "}", "(", ")", "exec",
               "set-location", "write-host", "write-output", "unset", "local", "declare", "read", "printf", "return", "exit", "q"}
_MULTI = {"npm", "pnpm", "yarn", "bun", "npx", "python", "python3", "py", "uv", "poetry", "dotnet", "go", "cargo", "git", "cmake", "node"}
# only these take a sub-command word; for every other verb (cat, grep, sed, ls ...) the next word is an argument and is never kept
_SUBCMD = _MULTI | {"docker", "kubectl", "gh", "az", "aws", "terraform", "pip", "pip3", "rustup", "make", "mvn", "gradle", "deno", "ruff",
                    "ollama", "claude", "conda", "brew", "winget", "choco", "helm", "systemctl", "pytest", "vitest", "jest", "playwright"}
_EXT = re.compile(r"(\.[A-Za-z0-9]{1,8})$")
_TEMPLATE = re.compile(r"(?i)(?<![\w-])\.env\.(?:example|sample|template|dist|defaults)\b")
# code, not a file: attribute access such as x.key / $_.Key / el.props.key / self.env.get(...)
_CODE_ATTR = re.compile(r"(?i)(?:\$_|\b[A-Za-z_]\w?|\b\w+\.\w+)\.key\b(?![.\w])|\.env\.(?:get|items|keys|copy|setdefault|pop)\b")
_LIST_VERBS = {"ls", "dir", "get-childitem", "gci", "test-path", "find", "stat", "wc", "rm", "del", "remove-item", "mkdir", "touch", "chmod",
               "attrib", "file", "du"}
_GIT_SAFE = {"status", "check-ignore", "ls-files", "add", "rm", "mv", "restore", "log"}

# test / build commands (verbs as produced by shell_verbs)
_TEST = re.compile(r"^(?:cargo (?:test|nextest)|pytest|py\.test|python3? -m (?:pytest|unittest)|py -m (?:pytest|unittest)|uv run(?: pytest)?|"
                   r"(?:npm|pnpm|yarn|bun) (?:test|run (?:test[\w:.-]*|e2e[\w:.-]*))|npx (?:vitest|jest|playwright|mocha)|vitest|jest|mocha|"
                   r"go test|dotnet test|mvn test|gradle test|node --test|deno test|ctest|make (?:test|check)|invoke-pester|tox|nox)\b")
_BUILD = re.compile(r"^(?:cargo (?:build|check|clippy|run)|tsc|npx tsc|(?:npm|pnpm|yarn|bun) run (?:build|lint|typecheck|check)[\w:.-]*|"
                    r"go (?:build|vet)|dotnet build|mvn (?:package|compile|verify)|gradle build|make|cmake --build|mypy|ruff(?: check)?|"
                    r"eslint|npx eslint|python3? -m (?:mypy|py_compile|compileall)|swift build|zig build)\b")

_EXIT = re.compile(r"^\s*Exit code (\d+)")
_CARGO = re.compile(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed")
_PASSED = re.compile(r"\b(\d+) passed\b")
_FAILED = re.compile(r"\b(\d+) (?:failed|failing)\b")
_DOTNET = re.compile(r"Failed:\s*(\d+),\s*Passed:\s*(\d+)")
_NODE_PASS = re.compile(r"(?m)(?:^|\s)(?:\u2139|#)\s*pass (\d+)")      # node --test / TAP summaries: "i pass 12", "# pass 12"
_NODE_FAIL = re.compile(r"(?m)(?:^|\s)(?:\u2139|#)\s*fail (\d+)")
_BACKGROUND = re.compile(r"Command running in background with ID|Running in the background|run_in_background")
_EXIT_ECHO = re.compile(r"(?mi)^\s*(?:exit(?:[ _]?code)?|rc|status)\s*[=: ]\s*([1-9]\d{0,2})\s*$")   # an echoed non-zero exit status
# shell redirects into data files (.json results, .log, .csv) are not code edits
_CODE_EXT = r"(?:py|rs|ts|tsx|js|jsx|mjs|cjs|go|java|kt|cs|rb|php|c|cc|cpp|h|hpp|swift|sql|sh|ps1|psm1|toml|ya?ml|vue|svelte|scala|lua|zig|tf|hcl|tfvars)"
_SHELL_WRITE = re.compile(r"(?i)>{1,2}\s*[\"']?[^\s\"'|;&<>]+\." + _CODE_EXT + r"\b|\bsed\s+(?:-[a-z]*i\b|--in-place)|\bperl\s+-[a-z]*i"
                          r"|\b(?:set-content|add-content|out-file)\b[^|;]*\." + _CODE_EXT + r"\b|\btee\s+(?:-a\s+)?[^\s|;&]+\." + _CODE_EXT + r"\b"
                          r"|\bgit\s+(?:apply|am|checkout\s+--|restore)\b|\bpatch\s+-p")
# positive evidence that a check PASSED (a silent, truncated or backgrounded check proves nothing)
_PASSMARK = re.compile(r"(?mi)^\s*OK(?: \(skipped=\d+\))?\s*$|\bFinished\b.*\btarget|\bbuilt in\b|\bcompiled successfully|\bBUILD SUCCESS"
                       r"|^\s*(?:exit|rc|status|exit code|tc|test)\s*[=: ]\s*0\s*$|\bfound 0 errors\b|\b0 errors?\b|\bno issues found\b"
                       r"|\ball checks passed\b|\blint_ok\b|^[.sx]+\s+\[100%\]\s*$|\bselftest ok\b|\b\d+ (?:tests?|pages|checks|files|specs) passed\b")
# a script (heredoc or -c/-e) that writes a code file: open(...,"w") / write_text / writeFileSync / Set-Content plus a code-file literal
_SCRIPT_WRITE = re.compile(r"write_text\(|\.write\(|open\([^)]*[\"'][wa]b?[\"']|writeFileSync|fs\.writeFile|set-content|out-file|writealltext")
_QUOTED_WRITE = re.compile(r"(?i)>{1,2}\s*[\"'][^\"']+\." + _CODE_EXT + r"[\"']")
_FAILMARK = re.compile(r"(?m)^error(?:\[E\d{4}\])?: |error TS\d{4}:|\bcould not compile\b|\bBuild FAILED\b|^npm ERR!|^SyntaxError: "
                       r"|^FAILED\b|^FAIL\s|--- FAIL:")


def _norm_name(name) -> str:
    name = str(name or "")
    return name if _NAME_OK.match(name) else "other"


def kind_of(name: str) -> str:
    if name in EDIT_TOOLS:
        return "edit"
    if name in READ_TOOLS:
        return "read"
    if name in SHELL_TOOLS:
        return "shell"
    if name in SEARCH_TOOLS:
        return "search"
    if name in WEB_TOOLS:
        return "web"
    if name in DISPATCH_TOOLS:
        return "dispatch"
    if name.startswith("mcp__"):
        return "mcp"
    return "other"


def arg_shape(inp) -> list[str]:
    """['command:str:2', 'timeout:num'] - key, value type and (for strings/lists) a size bucket. Keys only, never values."""
    if not isinstance(inp, dict):
        return []
    out = []
    for k in sorted(k for k in inp if isinstance(k, str))[:20]:
        if not _NAME_OK.match(k):
            continue
        v = inp[k]
        if isinstance(v, bool):
            t = "bool"
        elif isinstance(v, int | float):
            t = "num"
        elif isinstance(v, str):
            t = f"str:{int(math.log10(len(v))) if v else 0}"
        elif isinstance(v, list):
            t = f"list:{min(len(v), 9)}"
        elif isinstance(v, dict):
            t = "obj"
        else:
            t = "null"
        out.append(f"{k}:{t}")
    return out


def segments(cmd: str) -> list[str]:
    """Command segments split at && || ; | and newlines OUTSIDE quotes. Lines after a heredoc marker are dropped (file content)."""
    lines, body = [], False
    for line in str(cmd or "").splitlines():
        if body:
            continue
        lines.append(line)
        if "<<" in line:
            body = True
    text = "\n".join(lines)
    out, cur, q, i = [], [], None, 0
    while i < len(text):
        ch = text[i]
        two = text[i:i + 2]
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in "'\"":
            q = ch
            cur.append(ch)
        elif two in ("&&", "||"):
            out.append("".join(cur))
            cur = []
            i += 1
        elif ch in "\n;|":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [x for x in out if x.strip()]


def _unquoted(seg: str) -> str:
    """The segment with every quoted string replaced by a placeholder (quoted text is data: patterns, inline code, messages)."""
    return re.sub(r"\"(?:[^\"\\]|\\.)*\"|'[^']*'", " __q__ ", seg)


def _seg_tokens(seg: str) -> list[str]:
    toks = seg.strip().split()
    while toks and _ASSIGN.match(toks[0]):
        toks.pop(0)
    while toks and toks[0].lower() in ("sudo", "time", "nice", "timeout", "&", "!"):
        toks.pop(0)
        if toks and re.match(r"^\d+[smh]?$", toks[0]):
            toks.pop(0)
    return toks


def shell_verbs(cmd: str) -> list[str]:
    """Verbs of each command segment: 'cargo test', 'npm run build', 'python -m pytest', 'git status'. Quoted strings are never
    read (a quoted executable path contributes only its basename), heredoc bodies are ignored, and every word must match a
    restricted charset, else it is dropped."""
    if not cmd:
        return []
    out = []
    for seg in segments(cmd):
        first = re.match(r"\s*(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*&?\s*([\"'])([^\"']*)\1", seg)
        toks = _seg_tokens(_unquoted(seg))
        if first:
            toks = [first.group(2)] + toks[1:]
        if not toks:
            continue
        v = re.split(r"[\\/]", toks[0])[-1].lower().removesuffix(".exe")
        if v in _SKIP_VERBS or not _VERB_OK.match(v):
            continue
        words = [v]
        if v in _SUBCMD:
            for t in toks[1:3]:
                t = t.lower()
                ok = _RUN_ARG_OK.match(t) if words[-1] in ("run", "-m", "exec") else _SUB_OK.match(t)
                if not ok:
                    break
                words.append(t)
                if not (v in _MULTI and t in ("run", "-m", "--build", "exec")):
                    break
        verb = " ".join(words)
        if verb not in out:
            out.append(verb)
        if len(out) >= MAX_VERBS:
            break
    return [scrub(v).text for v in out]


def shell_writes_code(cmd: str) -> bool:
    """Does a shell command (outside heredoc bodies) write a code file: a redirect into *.py/*.rs/..., sed -i, Set-Content, tee,
    git apply/restore, patch? Only this boolean is kept. A python/node script that writes files is not detectable (known gap)."""
    return write_pos(cmd) >= 0


def write_pos(cmd: str) -> int:
    """Character offset of the LAST code-file write in a command (-1 = none): shell redirects/sed -i/tee/git apply outside quotes,
    redirects into quoted paths, and scripts (python heredoc, node -e, PowerShell) that write a file named with a code extension."""
    cmd = str(cmd or "")
    pos = -1
    segs = segments(cmd)
    for m in _SHELL_WRITE.finditer("\n".join(_unquoted(x) for x in segs)):
        pos = max(pos, _locate(cmd, m.group(0)))
    for m in _QUOTED_WRITE.finditer("\n".join(segs)):
        pos = max(pos, _locate(cmd, m.group(0)))
    if re.search(r"\." + _CODE_EXT + r"[\"']", cmd, re.I):
        for m in _SCRIPT_WRITE.finditer(cmd.lower()):
            pos = max(pos, m.start())
    return pos


def _locate(cmd: str, frag: str) -> int:
    i = cmd.lower().rfind(frag.strip().lower()[:12])
    return i if i >= 0 else 0


def check_pos(cmd: str) -> int:
    """Character offset of the LAST test/build command in a shell command (-1 = none)."""
    pos = -1
    low = str(cmd or "").lower()
    for m in re.finditer(r"\b(?:cargo (?:test|nextest|build|check|clippy|run)|pytest|py\.test|unittest|vitest|jest|playwright test|mocha"
                         r"|go (?:test|build|vet)|dotnet (?:test|build)|npm (?:test|run)|pnpm (?:test|run)|yarn (?:test|run)|node --test"
                         r"|tsc\b|mypy|ruff|eslint|make\b|mvn|gradle|ctest|deno test)", low):
        pos = max(pos, m.start())
    return pos


def check_kind(verbs: list[str]) -> str | None:
    if any(_TEST.match(v) for v in verbs):
        return "test"
    if any(_BUILD.match(v) for v in verbs):
        return "build"
    return None


def file_ext(inp) -> str | None:
    if not isinstance(inp, dict):
        return None
    p = inp.get("file_path") or inp.get("notebook_path") or inp.get("path")
    if not isinstance(p, str):
        return None
    m = _EXT.search(p.strip())
    return m.group(1).lower() if m else None


def call_taint(name: str, inp, path_text: str, verbs: list[str]) -> str | None:
    """None (no sensitive path), 'mention' (listing-only: no file content can reach the transcript), 'read' or 'write'.
    Template files (.env.example ...) and code attribute access (x.key, $_.Key, env.get) are not sensitive paths."""
    text = _CODE_ATTR.sub(" code ", _TEMPLATE.sub(".tmpl", path_text or ""))
    if not text or not SENSITIVE_PATH_RE.search(text):
        return None
    if name in ("Glob", "LS"):
        return "mention"
    if name == "Grep":
        mode = (inp or {}).get("output_mode") if isinstance(inp, dict) else None
        return "mention" if mode in (None, "files_with_matches", "count") else "read"
    if name in EDIT_TOOLS:
        return "write"
    if name in SHELL_TOOLS:
        cmd = _CODE_ATTR.sub(" code ", _TEMPLATE.sub(".tmpl", str((inp or {}).get("command") or "")))
        segs = [s for s in segments(cmd) if SENSITIVE_PATH_RE.search(s)]
        for s in segs:
            toks = _seg_tokens(_unquoted(s)) or _seg_tokens(s)
            if not toks:
                return "read"
            v = re.split(r"[\\/]", toks[0])[-1].lower().removesuffix(".exe")
            if v in _LIST_VERBS:
                continue
            if v == "git" and len(toks) > 1 and toks[1].lower() in _GIT_SAFE:
                continue
            return "read"
        return "mention" if segs else "read"
    return "read"


def result_summary(text: str, is_error: bool) -> dict:
    """Only ints / booleans from a tool result: ok, exit code, tests passed / failed, failure-marker flag."""
    text = text or ""
    r: dict = {"ok": not is_error}
    if _BACKGROUND.search(text[:400]):
        r["bg"] = True                      # the command was sent to the background: its real outcome is not in this result
        return r
    m = _EXIT.match(text)
    if m:
        r["ex"] = int(m.group(1))
    elif not is_error:
        r["ex"] = 0
    p = f = 0
    for a, b in _CARGO.findall(text):
        p += int(a)
        f += int(b)
    if not p and not f:
        p = sum(int(x) for x in _PASSED.findall(text))
        f = sum(int(x) for x in _FAILED.findall(text))
        for a, b in _DOTNET.findall(text):
            f += int(a)
            p += int(b)
        p += sum(int(x) for x in _NODE_PASS.findall(text))
        f += sum(int(x) for x in _NODE_FAIL.findall(text))
    if p or f:
        r["p"], r["f"] = p, f
    if _FAILMARK.search(text) or _EXIT_ECHO.search(_EXIT.sub("", text, count=1)):
        r["ce"] = True
    if _PASSMARK.search(text):
        r["pm"] = True
    return r


def check_pass(c: dict) -> bool | None:
    """Did a test/build call pass? None when its result never arrived."""
    if c.get("ok") is None or c.get("bg"):
        return None
    if not c["ok"] or c.get("f") or c.get("ce"):
        return False
    return True if (c.get("p") or c.get("pm")) else None     # no positive evidence (silent / truncated output): unknown


def is_code_edit(c: dict) -> bool:
    if c.get("we"):
        return True                         # a shell command that wrote a code file
    return c.get("k") == "edit" and (c.get("x") or "") not in DOC_EXT and c.get("ok") is not False


def evidence(calls: list[dict]) -> dict:
    """Execution evidence of a trajectory: was the last code edit followed by a passing test/build?"""
    last_edit = -1
    for i, c in enumerate(calls):
        if is_code_edit(c):
            last_edit = i
    edits = sum(1 for c in calls if is_code_edit(c))
    # a check counts when it ran after the last edit; a single shell call that edits and then tests ("sed -i ... && pytest") counts too
    checks = [c for i, c in enumerate(calls) if i >= max(last_edit, 0) and c.get("c") and check_pass(c) is not None
              and (i > last_edit or (c.get("we") and not c.get("wa")))]
    ev = {"edits": edits, "checks_after": len(checks), "n_calls": len(calls), "n_err": sum(1 for c in calls if c.get("ok") is False)}
    if not checks:
        ev.update(verified=None, why="no_check_after_last_edit" if edits else "no_edit_no_check")
        return ev
    last = checks[-1]
    ok = check_pass(last)
    ev.update(verified=ok, why=("last_check_pass" if ok else "last_check_fail"), check=last.get("c"),
              tests_passed=last.get("p"), tests_failed=last.get("f"))
    return ev


def tainted(calls: list[dict]) -> bool:
    """A trajectory is tainted when a sensitive file's CONTENT could have entered its context: a successful (or unknown) read,
    or any write. Listing-only mentions and template files are not taint; a failed Read returns an error, not the file."""
    for c in calls:
        t = c.get("tt")
        if t == "write":
            return True
        if t == "read" and not (c.get("n") in READ_TOOLS and c.get("ok") is False):
            return True
    return False


def profile(calls: list[dict]) -> dict:
    """Router features of a trajectory (counts only)."""
    kinds: dict = {}
    for c in calls:
        kinds[c.get("k", "other")] = kinds.get(c.get("k", "other"), 0) + 1
    ev = evidence(calls)
    return {"n_calls": len(calls), "n_err": ev["n_err"], "kinds": kinds, "edits": ev["edits"],
            "checks": sum(1 for c in calls if c.get("c")), "verified": ev["verified"]}


def compact(c: dict) -> dict:
    """A call record without the rationale text (what trajectories.jsonl stores)."""
    return {k: v for k, v in c.items() if k != "why"}


def action_text(c: dict) -> str:
    """'Bash[cargo test](command,description)', 'Edit[.rs](file_path,new_string,old_string)' - the next-action target of a step row."""
    tgt = ", ".join(c.get("v") or []) or (c.get("x") or "")
    keys = ",".join(s.split(":")[0] for s in c.get("sh") or [])
    return f"{c.get('n')}[{tgt}]({keys})"


def step_line(c: dict) -> str:
    res = "?" if c.get("ok") is None else ("ok" if c["ok"] else f"err{'' if c.get('ex') in (None, 0) else ' exit ' + str(c['ex'])}")
    if c.get("p") is not None or c.get("f") is not None:
        res += f" ({c.get('p', 0)} passed, {c.get('f', 0)} failed)"
    if c.get("ce"):
        res += " (failure marker)"
    return f"{action_text(c)} -> {res}"