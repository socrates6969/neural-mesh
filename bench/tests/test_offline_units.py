"""bench/tests/test_offline_units.py - offline unit tests for the parts the end-to-end tests do not reach: HELD/CONF rebuild
from the seeds, battery --check, the Claude/Ollama clients (subprocess / HTTP stubbed), the front-end's v2 and hook paths
(bench/frontend.py is imported, never modified), and common.py. No model, Ollama, docker or paid calls."""
import io, json, pathlib, subprocess, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import battery, common, frontend, run  # noqa: E402
from common import json_sha  # noqa: E402


# ---------------------------------------------------------------- battery.py
def test_held_conf_rebuild_from_seeds():
    """the committed HELD/CONF batteries are what the seeded templates + pre-registered leak check produce"""
    tune = battery.tune_items(); taken = {i["prompt"] for i in tune}
    with battery._FrontendConfig(battery.FRONTEND_ENV) as fp:
        pool = battery.const_pool()
        for s in ("held", "conf"):
            items, rep = battery.build_split(s, fp, pool, taken)
            taken |= {i["prompt"] for i in items}
            j = json.load(open(BENCH / f"battery_{s}.json", encoding="utf8"))
            assert rep["counts"] == j["header"]["counts"] and rep["shortfall"] == j["header"]["shortfall"]
            if json_sha(items) != j["header"]["items_sha"]:
                if frontend.corpus_sha() != j["header"]["corpus_sha"]:
                    pytest.skip(f"{s}: retrieval corpus changed since the build, so leak resampling may differ (battery --check warns)")
                pytest.fail(f"{s}: rebuild from seed {battery.SEEDS[s]} differs from the committed battery")
    assert frontend.V2 == (frontend.os.environ.get("MESH_V2", "0") == "1")             # _FrontendConfig restored the module


def test_build_split_details():
    with pytest.raises(ValueError):
        battery.build_split("tune")
    assert battery.split_of_const("x.rs", "A") in ("held", "conf")
    code = dict(kind="code", fn="f", answer=None)
    assert battery.answer_needles(code) == ["def f("]
    assert battery.answer_needles(dict(kind="str", answer="ab")) == [] and battery.answer_needles(dict(kind="str", answer="abcd")) == ["abcd"]
    assert battery.answer_needles(dict(kind="num", answer=42)) == [] and battery.answer_needles(dict(kind="num", answer=-1234)) == ["-1234"]
    assert "3.1416" in battery.answer_needles(dict(kind="num", answer=3.14159265))
    items, rep = battery.build_split("held", lambda p: (p, {"retrieved": False}), pool=[])          # no context -> no leak (resamples only for duplicates)
    assert rep["shortfall"] == {"research_private": 30} and len(items) == 120
    assert all(i["leak"] == "checked" for i in items)
    pool = [("x.rs", f"C{k}", str(k), "sha") for k in range(200)]
    priv, n = battery.private_items("conf", pool)
    assert len(priv) == 30 and n > 30 and all(p["sub"] == "private" and p["src_file"] == "src/x.rs" for p in priv)
    dup = lambda split, axis, idx, attempt: battery.make_item(split, axis, 0, 0)          # noqa: E731  (always the same prompt)
    with pytest.raises(battery.BuildError):
        battery.build_split("held", lambda p: (p, {"retrieved": False}), pool=[], make=dup)


def test_battery_check_mode(capsys):
    assert battery.main(["--check"]) == 0                                    # committed files are consistent with src/ and each other
    out = capsys.readouterr().out
    assert "check: OK" in out


def test_git_show_and_const_pool(tmp_path, monkeypatch):
    assert battery._git_show("src/definitely_not_a_file.rs") is None
    monkeypatch.setattr(battery.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert battery._git_show("src/x.rs") is None
    (tmp_path / "y.rs").write_text("const ONE: u32 = 1;\nconst ONE: u32 = 2;\npub(crate) const TAU: f64 = 6.28_3;\n", encoding="utf8")
    pool = battery.const_pool(tmp_path, committed_only=True)               # git unavailable: falls back to the working tree
    assert [(f, n, v) for f, n, v, _ in pool] == [("y.rs", "TAU", "6.283")]   # duplicated names are ambiguous and dropped


# ---------------------------------------------------------------- run.py model clients (stubbed transport)
class _P:
    def __init__(self, stdout="", stderr="", returncode=0): self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_call_claude(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "WORK", tmp_path / "cc")
    seen = {}
    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, env=kw["env"], cwd=kw["cwd"])
        return _P(json.dumps(dict(result="ANSWER: 4", is_error=False, stop_reason="end_turn", total_cost_usd=0.002, duration_api_ms=900,
                                  usage=dict(input_tokens=10, output_tokens=3, cache_read_input_tokens=5), modelUsage={"claude-haiku-x": {}})))
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    monkeypatch.setenv("BENCH_THINK", "1024")
    r = run.call_claude("2+2?")
    assert (r["cls"], r["text"], r["tokens"]["total"], r["model_ids"], r["cost_usd"]) == ("OK", "ANSWER: 4", 18, ["claude-haiku-x"], 0.002)
    assert seen["env"]["MAX_THINKING_TOKENS"] == "1024" and "--tools" in seen["cmd"] and seen["cmd"][seen["cmd"].index("--model") + 1] == run.CLAUDE_MODEL
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd, 240)))
    assert run.call_claude("x")["cls"] == "INFRA"
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(FileNotFoundError("claude")))
    r = run.call_claude("x")
    assert r["cls"] == "INFRA" and r["err"].startswith("spawn:") and r["tokens"] is None
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: _P("not json at all"))
    assert run.call_claude("x")["cls"] == "INFRA"


def test_call_ollama_and_digest(monkeypatch):
    class Resp(io.BytesIO): pass
    bodies = []
    def urlopen(req, timeout=None):
        if isinstance(req, str):           # /api/tags
            return Resp(json.dumps(dict(models=[dict(name="other"), dict(name=run.LOCAL_MODEL, digest="sha256:abc")])).encode())
        bodies.append(json.loads(req.data))
        return Resp(json.dumps(dict(response="ANSWER: 4", done_reason="stop", prompt_eval_count=7, eval_count=2, total_duration=3e9)).encode())
    monkeypatch.setattr(run.urllib.request, "urlopen", urlopen)
    r = run.call_ollama("2+2?")
    assert (r["cls"], r["text"], r["tokens"]["total"], r["api_ms"]) == ("OK", "ANSWER: 4", 9, 3000.0)
    assert bodies[0]["options"] == dict(temperature=0, seed=0, num_predict=1200) and bodies[0]["system"] == run.SYSTEM
    assert run.ollama_digest() == "sha256:abc" and run.ollama_digest("missing") is None
    monkeypatch.setattr(run.urllib.request, "urlopen", lambda req, timeout=None: Resp(json.dumps(dict(response="", done_reason="length")).encode()))
    assert run.call_ollama("x")["cls"] == "MODEL"
    def down(req, timeout=None): raise run.urllib.error.URLError("refused")
    monkeypatch.setattr(run.urllib.request, "urlopen", down)
    assert run.call_ollama("x")["cls"] == "INFRA" and run.ollama_digest() is None


def test_claude_cli_version(monkeypatch):
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: _P("2.9.0 (Claude Code)\n"))
    assert run.claude_cli_version() == "2.9.0 (Claude Code)"
    monkeypatch.setattr(run.subprocess, "run", lambda cmd, **kw: (_ for _ in ()).throw(OSError()))
    assert run.claude_cli_version() is None


def test_docker_unavailable_and_missing_battery(monkeypatch, tmp_path):
    monkeypatch.setattr(run.sandbox, "docker_available", lambda: (False, "daemon down"))
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    args = ["--split", "tune", "--cond", "A", "--tag", "dk", "--limit", "1", "--sandbox", "docker", "--no-analyze"]
    assert run.main(args) == run.EXIT_SANDBOX
    assert run.resolve_sandbox("auto", "tune") == "job" and run.resolve_sandbox("auto", "held") == "docker"
    monkeypatch.setattr(run, "battery_path", lambda s: tmp_path / "nope.json")
    assert run.main(args) == run.EXIT_PREREG
    with pytest.raises(SystemExit):
        run.parse(["--split", "held", "--tag", "x", "--limit", "2"])                  # --limit is TUNE-only


def test_plan_units_order_and_probe():
    items = [dict(id=f"i{k}") for k in range(3)]
    p = run.plan_units(items, ["A", "B", "C"], 2, done={("i0", "A", 0)}, probe_n=2)
    pairs = [x for x in p if x[0] == "pair"]
    assert ("one", items[0], "B", 0, False) in p and len(pairs) == 5                  # i0 rep0 only needs B
    assert {tuple(x[3]) for x in pairs} == {("A", "B"), ("B", "A")}                    # A/B order alternates
    assert [x[2:] for x in p if x[0] == "one" and x[2] == "C"] == [("C", 0, False)] * 3 + [("C", 1, True)] * 2


# ---------------------------------------------------------------- frontend.py (imported, never modified)
def test_frontend_v2_and_threshold(monkeypatch):
    hits, structured = frontend.retrieve_v2("what is `const MAX_AGENTS` in src/compact.rs?")
    assert structured and all(src.startswith("src/compact.rs:") for _, src, _ in hits) and "MAX_AGENTS" in hits[0][2]
    hits, structured = frontend.retrieve_v2("how does the router learn from outcomes")
    assert not structured and len(hits) == 3
    q = "In the neural-mesh Rust crate, what is the value of `const MAX_AGENTS` in src/compact.rs?" + battery.FMT
    assert frontend.query_of(q).endswith("src/compact.rs?")
    monkeypatch.setattr(frontend, "V2", True); monkeypatch.setattr(frontend, "HOOK_FE", False)
    new, meta = frontend.frontend_prompt(q)
    assert meta["retrieved"] and new.endswith(q) and "[src/compact.rs:" in new
    monkeypatch.setattr(frontend, "V2", False); monkeypatch.setattr(frontend, "THRESH", 1e9)
    assert frontend.frontend_prompt(q) == (q, {"retrieved": False, "top": frontend.frontend_prompt(q)[1]["top"]})
    assert set(frontend.frontend_env()) == {"MESH_V2", "MESH_THRESH", "MESH_FRONTEND"}


def test_frontend_hook_path(monkeypatch):
    calls = []
    def fake_run(cmd, input=None, **kw):
        calls.append((cmd, json.loads(input), kw["env"]["INGEST_DATA"]))
        return _P("<memory>the constant is 42</memory>\n" if "const" in input else "")
    monkeypatch.setattr(frontend.subprocess, "run", fake_run)
    monkeypatch.setattr(frontend, "HOOK_FE", True)
    new, meta = frontend.frontend_prompt("what is `const X`?")
    assert meta == {"retrieved": True, "hook": True, "hook_chars": len("<memory>the constant is 42</memory>")} and new.endswith("---\nwhat is `const X`?")
    assert frontend.frontend_prompt("hello") == ("hello", {"retrieved": False, "hook": True})
    cmd, payload, data = calls[0]
    assert cmd[-1].endswith("recall_hook.py") and payload["hook_event_name"] == "UserPromptSubmit"
    assert pathlib.Path(data, "memory", "items.jsonl").exists()                       # a frozen copy of the ledger, outside the repo
    assert not str(data).startswith(str(BENCH.parent))


# ---------------------------------------------------------------- common.py
def test_common_helpers(tmp_path, monkeypatch):
    header, items = common.load_battery("held")
    assert header["split"] == "held" and len(items) == 150
    assert common.load_prereg(tmp_path / "missing.json") == (None, None)
    (tmp_path / "a.txt").write_bytes(b"x\r\ny\n"); (tmp_path / "b.txt").write_bytes(b"x\ny\n")
    assert common.file_sha(tmp_path / "a.txt") == common.file_sha(tmp_path / "b.txt")      # CRLF-independent
    assert common.axis_of(dict(cat="research", sub="private")) == "research_private" and common.axis_of(dict(cat="code", sub="func")) == "code"
    gi = common.git_info()
    assert len(gi["git_sha"]) == 40 and isinstance(gi["git_dirty"], bool)
    monkeypatch.setattr(common.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    assert common.git_info() == dict(git_sha="", git_dirty=False)
