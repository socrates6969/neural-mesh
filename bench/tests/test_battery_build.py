"""bench/tests/test_battery_build.py - battery.py builds and checks every split in memory (leak check, disjointness) and only
then writes files; a failed check writes nothing. Offline; the real split files are never touched."""
import json, pathlib, sys
import pytest

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
import battery  # noqa: E402


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(battery, "write_split", lambda s, items, header: events.append(("write", s)))
    monkeypatch.setattr(battery, "write_legacy", lambda tune: events.append(("write", "battery.json")))
    monkeypatch.setattr(battery, "battery_path", lambda s: tmp_path / f"battery_{s}.json")      # nothing on disk for held/conf
    monkeypatch.setattr(battery, "const_pool", lambda *a, **k: [])
    real_check = battery.check_disjoint
    def check(splits):
        events.append(("check", tuple(sorted(splits)))); return real_check(splits)
    monkeypatch.setattr(battery, "check_disjoint", check)

    def fake_build(split, fp=None, pool=None, taken_prompts=(), make=None, max_resample=5):
        events.append(("build", split))
        it = dict(id=f"{split[0]}-ma01", split=split, cat="math", sub="exact", prompt=f"{split} prompt", answer=1, kind="num", tol=0, leak="checked")
        return [it], dict(counts={"math": 1}, shortfall={}, private_pool_for_split=0, resampled={}, private_retrieved=0)
    monkeypatch.setattr(battery, "build_split", fake_build)
    return events


def test_checks_run_before_any_write(recorder):
    assert battery.main(["--split", "held"]) == 0
    writes = [i for i, e in enumerate(recorder) if e[0] == "write"]
    checks = [i for i, e in enumerate(recorder) if e[0] == "check"]
    assert recorder[:2] == [("build", "held"), ("build", "conf")]              # both built (CONF avoids HELD prompts)
    assert checks and writes and max(checks) < min(writes)
    assert [e for e in recorder if e[0] == "write"] == [("write", "held")]       # only the requested split is written
    recorder.clear()
    assert battery.main(["--split", "tune"]) == 0
    assert recorder[0][0] == "check" and recorder[1:] == [("write", "battery.json"), ("write", "tune")]


def test_failed_disjointness_writes_nothing(recorder, monkeypatch):
    monkeypatch.setattr(battery, "check_disjoint", lambda splits: ["held/conf share 1 prompts"])
    for split in ("all", "tune", "held", "conf"):
        recorder.clear()
        assert battery.main(["--split", split]) == 1
        assert not [e for e in recorder if e[0] == "write"], (split, recorder)


def test_leak_in_conf_writes_no_held(recorder, monkeypatch):
    """CONF fails its leak check after HELD was built: HELD must not have been written either"""
    def build(split, *a, **k):
        recorder.append(("build", split))
        if split == "conf": raise battery.BuildError("conf math[1]: still leaking after 5 resamples")
        return [], dict(counts={}, shortfall={}, private_pool_for_split=0, resampled={}, private_retrieved=0)
    monkeypatch.setattr(battery, "build_split", build)
    assert battery.main(["--split", "held"]) == 1
    assert recorder == [("build", "held"), ("build", "conf")]


def test_tune_only_checked_against_splits_on_disk(recorder, tmp_path):
    held = json.load(open(BENCH / "battery_held.json", encoding="utf8"))
    tune = json.load(open(BENCH / "battery_tune.json", encoding="utf8"))["items"]
    planted = dict(held, items=held["items"] + [tune[0]])                       # a TUNE item copied into HELD on disk
    (tmp_path / "battery_held.json").write_text(json.dumps(planted), encoding="utf8")
    assert battery.main(["--split", "tune"]) == 1
    assert ("check", ("held", "tune")) in recorder and not [e for e in recorder if e[0] == "write"]
