"""Tests for train.memory. Importing this package marks the process (and its subprocesses) as a test run, so the live
data guard (config.guard_live) refuses the live workspace.db / card store under ANY runner, not only pytest."""
import os

os.environ["MESH_TEST"] = "1"
