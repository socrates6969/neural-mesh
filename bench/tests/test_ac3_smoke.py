"""bench/tests/test_ac3_smoke.py - AC-3, offline (no model, Ollama, docker or paid calls): a stubbed A/B/C/D model client drives
run.py --split tune --limit 2 --tag smoke-test into a temp results dir, then analyze: every row present and parsed, summary.json
written. The real front-end (B/D) and the real job-mode sandbox (code items, host-side scoring) run."""
import json, pathlib, shutil, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import analyze, run  # noqa: E402


def _stub_callers(items, calls):
    """a model that answers every TUNE item correctly in the strict format; B/D prompts carry retrieved context before the item"""
    def answer(prompt):
        it = next(i for i in items if prompt.endswith(i["prompt"]))
        calls.append(it["id"])
        if it["kind"] == "code": text = "Here you go.\n```python\n" + it["ref"] + "\n```"
        else: text = f"Working...\nANSWER: {it['answer']}"
        return dict(cls="OK", text=text, wall=0.01, api_ms=5, tokens=dict(input=10, output=5, total=15), cost_usd=0.0, stop="end_turn", model_ids=["stub"])
    return dict(claude=answer, ollama=answer)


def test_ac3_smoke(tmp_path, monkeypatch):
    items = json.load(open(BENCH / "battery_tune.json", encoding="utf8"))["items"]
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "claude_cli_version", lambda: "stub")
    monkeypatch.setattr(run, "ollama_digest", lambda m=None: "sha256:stub")
    calls = []
    rc = run.main(["--split", "tune", "--cond", "A,B,C,D", "--tag", "smoke-test", "--k", "1", "--limit", "2", "--sandbox", "job"],
                  callers=_stub_callers(items, calls), sleep=lambda s: None)
    assert rc == 0
    rows = [json.loads(l) for l in open(tmp_path / "smoke-test" / "raw.jsonl", encoding="utf8")]
    scored = [r for r in rows if not r["probe"]]
    # --limit 2 is per axis: 2 items x 5 axes (math, physics, code, research_public, research_private) x 4 conds = 40 scored
    # rows, plus the determinism probe (rep 1) of C and D on the same 10 items = 20 unscored rows
    assert len(scored) == 2 * 5 * 4 == 40 and len(rows) == 60 and len(calls) == 60
    assert {(r["axis"], r["cond"]) for r in scored} == {(a, c) for a in run.AXES for c in "ABCD"}
    assert all(r["status"] == "ok" and r["parsed"] is True and r["correct"] is True for r in rows)
    assert all(r["sbx"]["strict"]["status"] == "pass" for r in rows if r["cat"] == "code")       # real job-mode sandbox, host scoring
    assert any(r.get("retrieved") for r in rows if r["cond"] in "BD")                            # the real front-end ran for B/D
    s = json.load(open(tmp_path / "smoke-test" / "summary.json", encoding="utf8"))
    assert s["splits"]["tune"]["n_items"] == 10 and "gate" not in s                              # TUNE: descriptive only
    assert s["splits"]["tune"]["describe"]["claude"]["ALL"]["diff"] == 0.0
    assert (tmp_path / "smoke-test" / "summary.txt").exists() and (tmp_path / "smoke-test" / "manifest_tune.json").exists()
    assert "TUNE-only descriptive evidence" in s["rd_log_line"]
