"""Test isolation for mesh_mcp: no test may ever touch the live Global Workspace database.

Runs before any test module imports the backend: points train.memory.workspace at a throwaway DB and key
(MESH_WORKSPACE_DB / MESH_WORKSPACE_KEY are read once at import) and wraps workspace_tools._workspace so that a
backend whose DB resolves outside this temp dir fails the test run loudly instead of writing to it.
"""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="meshmcp-tests-"))
atexit.register(shutil.rmtree, TMP, True)
os.environ["MESH_WORKSPACE_DB"] = str(TMP / "workspace.db")
os.environ["MESH_WORKSPACE_KEY"] = str(TMP / "workspace.key")
os.environ["MESH_REPORTS_DIR"] = str(TMP / "reports")
os.environ["MESH_PIPELINE_KEY"] = str(TMP / "pipeline.key")
os.environ["NEURAL_MESH_MEMORY"] = str(TMP / "mem")  # card store, usefulness and injection log
os.environ["MESH_TEST"] = "1"  # the backend itself refuses the live DB / card store when this is set
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


def _install_guard():
    from tools.mesh_mcp import workspace_tools
    real = workspace_tools._workspace

    def guarded():
        mod, err = real()
        if mod is not None:
            check_backend(mod)
        return mod, err

    workspace_tools._workspace = guarded


_install_guard()
