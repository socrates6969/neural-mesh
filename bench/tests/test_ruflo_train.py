"""bench/tests/test_ruflo_train.py - runs the node tests of bench/ruflo_train.mjs (a ruflo CLI failure is recorded as an error,
never counted as a wrong route). Stubbed CLI only; skipped when node is not installed."""
import pathlib, shutil, subprocess
import pytest

TESTS = pathlib.Path(__file__).resolve().parent


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_ruflo_train_node_suite():
    p = subprocess.run(["node", "--test", str(TESTS / "ruflo_train.test.mjs")], capture_output=True, text=True, encoding="utf8", errors="replace", timeout=120)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "# fail 0" in p.stdout or "fail 0" in p.stdout
