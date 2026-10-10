"""bench/tests/test_ac2_v2.py - AC-2, offline: the committed bench/results/v2/raw.jsonl reproduces the v2 result through
analyze.py: B-A ALL +0.117 [+0.050, +0.183], research:private B-A +1.000 [+1.000, +1.000] (and the committed summary.txt)."""
import json, pathlib, shutil, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import analyze, run  # noqa: E402


def test_ac2_v2_reproduces(tmp_path, monkeypatch):
    src = BENCH / "results" / "v2" / "raw.jsonl"
    assert src.exists(), "bench/results/v2/raw.jsonl is committed evidence"
    (tmp_path / "v2").mkdir()
    shutil.copy(src, tmp_path / "v2" / "raw.jsonl")                 # analyze writes summary.* next to raw.jsonl: never into the repo
    monkeypatch.setattr(analyze, "RESULTS", tmp_path)
    assert analyze.main(["v2"]) == 0
    s = json.load(open(tmp_path / "v2" / "summary.json", encoding="utf8"))
    all_b = s["ALL"]["paired_vs_A"]["B"]
    assert all_b["n"] == 120
    assert [round(x, 3) for x in all_b["acc"]] == [0.117, 0.050, 0.183]
    priv = s["research:private"]["paired_vs_A"]["B"]
    assert priv["n"] == 15 and priv["acc"] == [1.0, 1.0, 1.0]
    assert (s["research:private"]["A"]["acc"], s["research:private"]["B"]["acc"]) == (0.0, 1.0)
    txt = (tmp_path / "v2" / "summary.txt").read_text(encoding="utf8")
    assert "B-A (n=120): acc +0.117 [+0.050,+0.183]" in txt
    committed = (BENCH / "results" / "v2" / "summary.txt").read_text(encoding="utf8").replace("\r\n", "\n")
    assert txt.replace("\r\n", "\n") == committed                     # the whole committed summary, byte for byte
