"""bench/tests/test_bench.py - offline tests for the bench battery (no model calls, no network except a local docker daemon).
Run: python -m pytest bench/tests -q"""
import json, os, pathlib, random, sys, time
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import sandbox  # noqa: E402

DOCKER_OK, DOCKER_WHY = sandbox.docker_available()
needs_docker = pytest.mark.skipif(not DOCKER_OK, reason=f"docker sandbox unavailable: {DOCKER_WHY}")

ESCAPES = {   # candidate -> the value it returns when it is CONTAINED
    "file_write": ("def f():\n    try:\n        open('/etc/bench_pwned', 'w').write('x')\n        open('pwned.txt', 'w').write('x')\n        return 'wrote'\n"
                   "    except Exception:\n        return 'blocked'", "blocked"),
    "socket": ("def f():\n    import socket\n    try:\n        socket.create_connection(('1.1.1.1', 53), 2)\n        return 'connected'\n"
               "    except Exception:\n        return 'blocked'", "blocked"),
    "fork_bomb": ("def f():\n    import os, time\n    n = 0\n    try:\n        while n < 5000:\n            if os.fork() == 0:\n                time.sleep(60); os._exit(0)\n"
                  "            n += 1\n    except OSError:\n        return 'limited'\n    return 'unlimited'", "limited"),
    "infinite_loop": ("def f():\n    while True:\n        pass", None),
}


@needs_docker
@pytest.mark.parametrize("name", list(ESCAPES))
def test_sandbox_contains(name):
    code, contained = ESCAPES[name]
    t0 = time.time()
    r = sandbox.run_code(code, "f", [[]], [sandbox.canon(contained)], mode="docker")
    assert time.time() - t0 < 25, r
    if name == "infinite_loop":
        assert r["status"] == "timeout", r
    else:
        assert r["status"] == "pass", r          # pass == the candidate observed it was contained
    assert not (BENCH / "pwned.txt").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="job-object fallback is Windows-specific")
def test_sandbox_job_fallback_basic():
    """the TUNE-only fallback: right/wrong answers, an audit-hook refusal and the host timeout (NOT a security boundary)"""
    assert sandbox.run_code("def f(x): return x * 2", "f", [[3]], [sandbox.canon(6)], mode="job")["status"] == "pass"
    assert sandbox.run_code("def f(x): return x", "f", [[3]], [sandbox.canon(6)], mode="job")["status"] == "fail"
    code, contained = ESCAPES["file_write"]
    assert sandbox.run_code(code, "f", [[]], [sandbox.canon(contained)], mode="job")["status"] == "pass"
    assert sandbox.run_code(ESCAPES["infinite_loop"][0], "f", [[]], [sandbox.canon(None)], mode="job", timeout=3)["status"] == "timeout"


# ====================================================================== run.py
import run  # noqa: E402

NOSLEEP = lambda s: None  # noqa: E731


def _cj(text="ANSWER: 4", stop="end_turn", is_error=False):
    return json.dumps(dict(type="result", result=text, is_error=is_error, stop_reason=stop, usage=dict(input_tokens=10, output_tokens=5),
                           duration_api_ms=100, total_cost_usd=0.001, modelUsage={"claude-haiku-test": {}}))


def _const(cls, text="ANSWER: 4"):
    return lambda prompt: dict(cls=cls, text=text, wall=0.1, tokens=dict(total=1), err=None if cls == "OK" else "x")


def test_classify(monkeypatch):
    c = run.classify_claude
    assert c(False, _cj())[0] == "OK"
    assert c(True, "")[0] == "INFRA"                                          # host timeout
    assert c(False, "Error: socket hang up")[0] == "INFRA"                    # not JSON
    assert c(False, _cj("API Error: 529 Overloaded", is_error=True))[0] == "INFRA"
    assert c(False, _cj("Claude AI usage limit reached", is_error=True))[0] == "INFRA"
    assert c(False, _cj("", is_error=True))[0] == "INFRA"
    assert c(False, _cj("I cannot help with that", is_error=True))[0] == "MODEL"
    assert c(False, _cj("partial", stop="max_tokens"))[0] == "MODEL"
    assert c(False, _cj("   "))[0] == "MODEL"
    item = dict(id="x", cat="math", sub="exact", prompt="2+2?", answer=4, kind="num", tol=0)
    rows = {cls: run.run_unit(item, "A", 0, "tune", "job", max_tries=2, sleep=NOSLEEP, callers=dict(claude=_const(cls), ollama=_const(cls)))
            for cls in ("OK", "MODEL", "INFRA")}
    assert rows["OK"]["status"] == "ok" and rows["OK"]["correct"] is True
    assert rows["MODEL"]["status"] == "model_fail" and rows["MODEL"]["correct"] is False        # a model failure scores as wrong
    assert rows["INFRA"]["status"] == "infra_fail" and rows["INFRA"]["correct"] is None and rows["INFRA"]["attempts"] == 2   # never scored
    code = dict(id="c", cat="code", sub="func", prompt="p", kind="code", answer=None, fn="f", tests=[[1]], expected=["2"])
    monkeypatch.setattr(run.sandbox, "run_code", lambda *a, **k: dict(status="infra", why="daemon down"))
    r = run.run_unit(code, "A", 0, "tune", "docker", max_tries=2, sleep=NOSLEEP, callers=dict(claude=_const("OK", "```python\ndef f(x): return 2*x\n```")))
    assert r["status"] == "infra_fail" and r["correct"] is None                 # a sandbox outage is infra, not a wrong answer


def test_retry():
    sleeps = []
    seq = iter(["INFRA", "INFRA", "OK"])
    r, n, errs = run.call_with_retry(lambda: dict(cls=next(seq), err="boom"), max_tries=5, sleep=sleeps.append, rng=random.Random(0))
    assert (r["cls"], n, len(sleeps), len(errs)) == ("OK", 3, 2, 2)
    assert 0 <= sleeps[0] <= 2.0 and 0 <= sleeps[1] <= 4.0                     # full jitter under an exponential cap
    sleeps.clear()
    r, n, _ = run.call_with_retry(lambda: dict(cls="INFRA", err="x"), max_tries=5, sleep=sleeps.append)
    assert (r["cls"], n, len(sleeps)) == ("INFRA", 5, 4)
    r, n, _ = run.call_with_retry(lambda: dict(cls="MODEL"), max_tries=5, sleep=sleeps.append)
    assert (r["cls"], n) == ("MODEL", 1)                                       # model failures are not retried

    class Top:
        @staticmethod
        def uniform(a, b): return b
    assert [run.backoff(k, rng=Top) for k in range(1, 8)] == [2, 4, 8, 16, 32, 60, 60]


def test_strict_parse():
    s = run.extract_strict
    assert s("work...\nANSWER: 42") == "42"
    assert s("**ANSWER:** 42\n\n") == "42"
    assert s("ANSWER: `abc`.") == "abc"
    assert s("ANSWER: 41\nactually 42") == ""                                  # not on the last line
    assert s("the answer is 42") == ""
    num = dict(kind="num", answer=42, tol=0)
    r = run.score(num, "I think it is 42")
    assert r["correct"] is False and r["fallback_correct"] is True and r["parsed"] is False   # fallback is diagnostic only
    r = run.score(num, "ANSWER: 42")
    assert r["correct"] is True and r["parsed"] is True
    assert run.code_of("```python\ndef f(): pass\n```") == "def f(): pass\n"
    assert run.code_of("def f(): pass") is None and run.code_of("def f(): pass", strict=False) == "def f(): pass"


def _mini_battery(tmp_path, split="tune", n=3):
    items = [dict(id=f"m{i}", split=split, cat="math", sub="exact", prompt=f"{i}+1?", answer=i + 1, kind="num", tol=0) for i in range(n)]
    p = tmp_path / f"battery_{split}.json"
    p.write_text(json.dumps(dict(header=dict(split=split, items_sha=run.json_sha(items)), items=items)), encoding="utf8")
    return p, items


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    bp, items = _mini_battery(tmp_path)
    monkeypatch.setattr(run, "battery_path", lambda s: bp)
    monkeypatch.setattr(run, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run, "stale_check", lambda *a: None)
    monkeypatch.setattr(run, "claude_cli_version", lambda: "test")
    monkeypatch.setattr(run, "ollama_digest", lambda m=None: "sha256:test")
    monkeypatch.setattr(run, "frontend_prompt", lambda p: (p, {"retrieved": False}))
    for k in run.ENV_KEYS: monkeypatch.delenv(k, raising=False)
    return tmp_path, items


def _fake(fail_ids=(), calls=None):
    def claude(prompt):
        if calls is not None: calls.append(prompt)
        if any(prompt.startswith(f"{i}+") for i in fail_ids): return dict(cls="INFRA", text="", wall=0.0, err="503", tokens=None)
        return dict(cls="OK", text=f"ANSWER: {int(prompt.split('+')[0]) + 1}", wall=0.01, tokens=dict(total=7), cost_usd=0.0)
    return dict(claude=claude, ollama=claude)


def _rows(tmp_path, tag):
    return [json.loads(l) for l in open(tmp_path / "results" / tag / "raw.jsonl", encoding="utf8")]


def test_resume_no_dupes(offline_run):
    tmp_path, items = offline_run
    args = ["--split", "tune", "--cond", "A,B", "--tag", "t", "--k", "2", "--sandbox", "job", "--max-tries", "2", "--no-analyze"]
    assert run.main(args, callers=_fake(fail_ids=[1]), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    assert sum(r["status"] == "infra_fail" for r in _rows(tmp_path, "t")) == 4          # item 1: 2 conds x 2 reps, written but unscored
    calls = []
    assert run.main(args, callers=_fake(calls=calls), sleep=NOSLEEP) == run.EXIT_OK
    assert len(calls) == 4                                                              # only the infra_fail units were re-run
    final = [(r["id"], r["cond"], r["rep"]) for r in _rows(tmp_path, "t") if r["status"] in ("ok", "model_fail")]
    assert len(final) == len(set(final)) == 3 * 2 * 2                                   # one final row per (id, cond, rep)
    assert run.main(args, callers=_fake(calls=calls), sleep=NOSLEEP) == run.EXIT_OK and len(calls) == 4   # nothing left to do
    (tmp_path / "results" / "t" / ".lock").write_text("{}")                            # a second writer
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_LOCK
    with pytest.raises(SystemExit) as e:
        run.parse(["--split", "tune", "--tag", "t", "--cond", "A,E"])
    assert e.value.code == run.EXIT_USAGE


def test_manifest_mismatch(offline_run, monkeypatch):
    tmp_path, items = offline_run
    args = ["--split", "tune", "--cond", "A", "--tag", "m", "--k", "1", "--sandbox", "job", "--no-analyze"]
    assert run.main(args, callers=_fake(fail_ids=[0]), sleep=NOSLEEP) == run.EXIT_INCOMPLETE
    man = json.load(open(tmp_path / "results" / "m" / "manifest_tune.json"))
    assert {"git_sha", "git_dirty", "battery_sha", "corpus_sha", "frontend_sha", "prereg_sha", "models", "K", "sandbox", "env"} <= set(man)
    monkeypatch.setenv("BENCH_THINK", "9999")                                   # a different thinking budget is a different experiment
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_PREREG
    monkeypatch.delenv("BENCH_THINK")
    assert run.main(args[:7] + ["2"] + args[8:], callers=_fake(), sleep=NOSLEEP) == run.EXIT_PREREG     # K changed
    assert run.main(args, callers=_fake(), sleep=NOSLEEP) == run.EXIT_OK


# ====================================================================== battery.py
import battery  # noqa: E402
from common import axis_of, file_sha, json_sha  # noqa: E402


def test_tune_identical():
    """the TUNE build reproduces today's frozen bench/battery.json byte for byte (line endings normalised for autocrlf)"""
    fresh = json.dumps(battery.tune_items(), indent=1).encode()
    on_disk = (BENCH / "battery.json").read_bytes().replace(b"\r\n", b"\n")
    assert fresh == on_disk
    j = json.load(open(BENCH / "battery_tune.json", encoding="utf8"))
    assert json_sha(j["items"]) == j["header"]["items_sha"] and len(j["items"]) == 120
    assert [dict((k, v) for k, v in i.items() if k not in ("split", "expected", "src_file", "const", "leak")) for i in j["items"]] == json.loads(json.dumps(battery.tune_items()))


def test_splits_disjoint():
    splits = {s: json.load(open(BENCH / f"battery_{s}.json", encoding="utf8"))["items"] for s in ("tune", "held", "conf")}
    assert battery.check_disjoint(splits) == []
    for s in ("held", "conf"):
        j = json.load(open(BENCH / f"battery_{s}.json", encoding="utf8"))
        assert json_sha(j["items"]) == j["header"]["items_sha"]
        counts = {}
        for i in j["items"]: counts[axis_of(i)] = counts.get(axis_of(i), 0) + 1
        assert counts == {a: 30 - j["header"]["shortfall"].get(a, 0) for a in ("math", "physics", "code", "research_public", "research_private")}
        assert all(i["leak"] == ("by-construction" if i["sub"] == "private" else "checked") for i in j["items"])
        assert all(len(i["tests"]) == len(i["expected"]) == 8 for i in j["items"] if i["kind"] == "code")   # 4 fixed + 4 rng inputs
        assert not any("errno" in i["prompt"] for i in j["items"])
    tune_consts = {(f"src/{f}.rs", n) for f, n in battery.TUNE_CONSTS}
    for s in ("held", "conf"):
        assert not tune_consts & {(i["src_file"], i["const"]) for i in splits[s] if i["sub"] == "private"}
    planted = dict(splits, conf=splits["conf"] + [splits["held"][0]])
    assert battery.check_disjoint(planted)                                      # an overlap is caught


def _leaky_fp(needle_always=True, first_only=False):
    seen = set()
    def fp(prompt):
        if first_only and prompt in seen: return prompt, {"retrieved": False}
        seen.add(prompt)
        return "Context retrieved from the project repository:\nANSWER_IN_CORPUS\n---\n" + prompt, {"retrieved": True, "top": 99.0}
    return fp


def test_leak_planted(monkeypatch):
    item = dict(id="h-ma01", split="held", cat="math", sub="exact", prompt="What is 123456 + 0?" + battery.FMT, answer=123456, kind="num", tol=0)
    ctx_with_answer = lambda p: ("[DESIGN.md:1]\nthe magic number is 123456\n\n---\n" + p, {"retrieved": True})
    ctx_without = lambda p: ("[DESIGN.md:1]\nnothing to see\n\n---\n" + p, {"retrieved": True})
    assert battery.leaks(item, ctx_with_answer) and not battery.leaks(item, ctx_without)
    assert not battery.leaks(item, lambda p: (p, {"retrieved": False}))          # nothing attached, nothing leaked
    calls = []
    def make(split, axis, idx, attempt):
        calls.append(attempt); return dict(item, prompt=f"{item['prompt']} [{axis}{idx}/{attempt}]")
    flaky = lambda p: ctx_with_answer(p) if p.endswith(("/0]", "/0]" + battery.FMT)) or "/0]" in p else ctx_without(p)
    items, rep = battery.build_split("held", flaky, pool=[], make=make)
    assert rep["resampled"] and all("/0]" not in i["prompt"] for i in items)     # every first draw leaked -> resampled once
    with pytest.raises(battery.BuildError):
        battery.build_split("held", ctx_with_answer, pool=[], make=make)         # leaks on every draw -> give up
    monkeypatch.setattr(battery, "leaks", lambda it, fp: True)
    monkeypatch.setattr(battery, "write_split", lambda *a: pytest.fail("must not write a leaking battery"))
    assert battery.main(["--split", "held"]) == 1                               # exit 1


def test_stale_const(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    rs = tmp_path / "src" / "x.rs"
    rs.write_text("pub const ALPHA: f64 = 0.25;\nconst BETA: usize = 1_000;\n", encoding="utf8")
    monkeypatch.setattr(battery, "NM", tmp_path)
    pool = battery.const_pool(tmp_path / "src", committed_only=False)
    assert {(f, n, v) for f, n, v, _ in pool} == {("x.rs", "ALPHA", "0.25"), ("x.rs", "BETA", "1000")}
    items = [dict(id="h-rv01", sub="private", src_file="src/x.rs", const="BETA", answer="1000", prompt="`const BETA`")]
    header = dict(corpus_sha="not-the-current-corpus", frontend_sha=None)
    probs = battery.check_items("held", header, items)
    assert [p[0] for p in probs] == ["warn"]                                     # a corpus change only warns
    rs.write_text("pub const ALPHA: f64 = 0.25;\nconst BETA: usize = 2_000;\n", encoding="utf8")
    assert "stale" in [p[0] for p in battery.check_items("held", header, items)]
    rs.write_text("pub const ALPHA: f64 = 0.25;\n", encoding="utf8")
    assert "stale" in [p[0] for p in battery.check_items("held", header, items)]  # vanished
    bp = tmp_path / "battery_held.json"
    bp.write_text(json.dumps(dict(header=dict(header, items_sha=json_sha(items)), items=items)), encoding="utf8")
    monkeypatch.setattr(battery, "battery_path", lambda s: bp)
    assert battery.main(["--check", "--split", "held"]) == 2                    # exit 2
    monkeypatch.setattr(run, "battery_path", lambda s: bp)
    monkeypatch.setattr(battery, "battery_path", lambda s: bp)
    assert run.main(["--split", "held", "--tag", "x", "--dry-run", "--sandbox", "docker"]) == run.EXIT_PREREG


# ====================================================================== analyze.py
import analyze  # noqa: E402
import lib as b1lib  # noqa: E402  (py/b1/lib.py, put on sys.path by analyze)


def _gate_fixture(tmp_path, monkeypatch, regress=False, n=6, K=2):
    res = tmp_path / "results"; (res / "G").mkdir(parents=True)
    paths = {}
    rows = []
    for split in ("held", "conf"):
        items = []
        for ax in ("math", "physics", "code", "research_public", "research_private"):
            cat, sub = ("research", ax.split("_")[1]) if ax.startswith("research") else (ax, "x")
            items += [dict(id=f"{split[0]}-{ax}{i}", split=split, cat=cat, sub=sub, prompt=f"{split}{ax}{i}", answer=1, kind="num", tol=0) for i in range(n)]
        p = tmp_path / f"battery_{split}.json"; p.write_text(json.dumps(dict(header={}, items=items)), encoding="utf8"); paths[split] = p
        (res / "G" / f"manifest_{split}.json").write_text(json.dumps(dict(split=split, conds=["A", "B"], K=dict(claude=K, local=1), battery_sha=json_sha(items))))
        for it in items:
            ax = axis_of(it); i = int(it["id"][-1])
            for rep in range(K):
                a = 0.0 if ax == "research_private" else float(i % 2 == 0)
                b = 1.0 if ax == "research_private" else a
                if regress and ax == "math": a, b = 1.0, 0.0                       # a consistent guard-axis loss
                for cond, c in (("A", a), ("B", b)):
                    rows.append(dict(split=split, id=it["id"], cat=it["cat"], sub=it["sub"], axis=ax, cond=cond, rep=rep, probe=False, status="ok",
                                     correct=bool(c), fallback_correct=bool(c), parsed=True, tokens=dict(total=100 + 50 * (cond == "B")), wall_s=1.0, attempts=1, text=""))
    monkeypatch.setattr(analyze, "RESULTS", res)
    monkeypatch.setattr(analyze, "battery_path", lambda s: paths[s])
    monkeypatch.setattr(analyze, "load_prereg", lambda: (dict(tol=0.002, delta=0.01), "f" * 64))
    def write(rs):
        (res / "G" / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf8")
    write(rows)
    return res, rows, write


def test_gate_reuse(tmp_path, monkeypatch):
    assert analyze.holm is b1lib.holm and analyze.tost is b1lib.tost and analyze.gate.compare.__module__ == "gate"   # reused, not reimplemented
    res, rows, write = _gate_fixture(tmp_path, monkeypatch)
    assert analyze.main(["G"]) == 0
    s = json.load(open(res / "G" / "summary.json"))
    assert s["gate"]["verdict"] == "KEEP" and s["gate"]["holm_family"] == 10          # 5 axes x 1 world x 2 splits
    assert "KEEP" in s["rd_log_line"] and "not gated" in s["rd_log_line"]
    res, rows, write = _gate_fixture(tmp_path / "r", monkeypatch, regress=True)
    assert analyze.main(["G"]) == 0
    s = json.load(open(res / "G" / "summary.json"))
    assert s["gate"]["verdict"] == "REJECT-regress" and any(l[0] == "math" for l in s["gate"]["losses"])
    write([r for r in rows if not (r["id"] == "h-math0" and r["cond"] == "B" and r["rep"] == 1)])     # one half of a pair missing
    assert analyze.main(["G"]) == 3
    assert analyze.main(["G", "--drop-incomplete"]) == 0
    assert json.load(open(res / "G" / "summary.json"))["splits"]["held"]["dropped"] == {"claude": ["h-math0"]}
    write(rows + [rows[0]])                                                            # a duplicate final row
    assert analyze.main(["G"]) == 4
    write([r for r in rows if r["split"] == "held"])
    assert analyze.main(["G"]) == 0
    assert json.load(open(res / "G" / "summary.json"))["gate"]["verdict"].startswith("PARTIAL")
