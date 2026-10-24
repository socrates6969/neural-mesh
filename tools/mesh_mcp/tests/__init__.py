r"""Test isolation for mesh_mcp: no test may ever touch the live Global Workspace database or card store.

Runs before any test module uses the backend:
- points train.memory.* at a throwaway dir (DB, keys, card store; these env vars are read at import time),
- if another suite in this process already imported the backend with live paths, reloads those modules,
- wraps workspace_tools._workspace so a backend whose DB resolves outside this dir fails the run loudly.
pytest may import this package twice (as tools.mesh_mcp.tests and mesh_mcp.tests); both share one dir and guard.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

if os.environ.get("MESH_MCP_TEST_TMP"):  # second import of this package in the same process: share the dir
    TMP = Path(os.environ["MESH_MCP_TEST_TMP"])
else:
    TMP = Path(tempfile.mkdtemp(prefix="meshmcp-tests-"))
    os.environ["MESH_MCP_TEST_TMP"] = str(TMP)
    atexit.register(shutil.rmtree, TMP, True)
os.environ["MESH_WORKSPACE_DB"] = str(TMP / "workspace.db")
os.environ["MESH_WORKSPACE_KEY"] = str(TMP / "workspace.key")
os.environ["MESH_REPORTS_DIR"] = str(TMP / "reports")
os.environ["MESH_PIPELINE_KEY"] = os.environ["NEURAL_MESH_PIPELINE_KEY"] = str(TMP / "pipeline.key")
os.environ["NEURAL_MESH_MEMORY"] = str(TMP / "mem")  # card store, usefulness and injection log
os.environ["NEURAL_MESH_MEMORY_KEY"] = str(TMP / "mem.key")
os.environ["MESH_TEST"] = "1"  # the backend itself refuses the live DB / card store when this is set
# use this checkout's own backend, never the default C:\Users\mariu\neural-mesh (it may be a different version)
os.environ.setdefault("MESH_WORKSPACE_ROOT", str(Path(__file__).resolve().parents[3]))
LIVE_DB = Path.home() / "neural-mesh" / "data" / "lifelong" / "mesh" / "workspace.db"


class LiveDatabaseError(AssertionError):
    pass


def check_backend(mod) -> None:
    """Raise if a workspace backend module would read or write a database outside the test temp dir."""
    db = getattr(mod, "DB", None)
    if db is None:
        return  # fake test backends have no DB
    db = Path(db).resolve()
    if db == LIVE_DB.resolve() or TMP.resolve() not in db.parents:
        raise LiveDatabaseError("refusing to run: workspace backend resolves to %s (tests must use %s)" % (db, TMP))


def _reload_backend() -> None:
    """Re-execute already-imported train.memory modules so their module-level paths pick up the env above:
    config first (others derive paths from it), then trust, then the rest. reload() keeps the module objects,
    so `from . import config as C` references elsewhere see the new values."""
    import importlib
    names = [n for n in list(sys.modules) if n.startswith("train.memory.") and sys.modules[n] is not None
             and ".tests" not in n and not n.endswith("__main__")]
    order = {"train.memory.config": 0, "train.memory.trust": 1}
    for n in sorted(names, key=lambda n: (order.get(n, 2), n)):
        try:
            importlib.reload(sys.modules[n])
        except Exception as e:  # noqa: BLE001
            raise LiveDatabaseError("could not re-point %s at the test dir: %s" % (n, e)) from e


def _install_guard() -> None:
    from tools.mesh_mcp import workspace_tools
    if getattr(workspace_tools._workspace, "_mesh_guard", False):
        return  # already installed by the other import of this package
    real = workspace_tools._workspace

    def guarded():
        mod, err = real()
        if mod is not None:
            check_backend(mod)
        return mod, err

    guarded._mesh_guard = True
    workspace_tools._workspace = guarded


_reload_backend()
_install_guard()
if sys.modules.get("train.memory.workspace") is not None:
    check_backend(sys.modules["train.memory.workspace"])  # fail at collection time if the re-point did not work
