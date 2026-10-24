"""Server protocol, output cap, report formatter and workspace wrapper tests (no network, no Docker)."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from tools.mesh_mcp import report, server, workspace_tools
from tools.mesh_mcp.common import MAX_OUT, as_json, cap

EXPECTED = {"literature_search", "math_solve", "stats_test", "causal_analysis", "discover_graph", "discover_dag_notears",
            "invariant_causal_prediction", "phack_guard", "market_series", "code_run",
            "workspace_post", "workspace_read", "memory_search", "analogies", "workspace_observe",
            "workspace_stats", "lean_check", "report_imrad"}


def rpc(method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


class ProtocolTest(unittest.TestCase):
    def test_initialize_echoes_protocol_version(self):
        r = rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]
        self.assertEqual(r["protocolVersion"], "2025-06-18")
        self.assertEqual(r["serverInfo"]["name"], "mesh-tools")

    def test_tools_list_has_all_tools_with_schemas(self):
        tools = rpc("tools/list")["result"]["tools"]
        self.assertEqual({t["name"] for t in tools}, EXPECTED)
        for t in tools:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertNotIn("fn", t)
            self.assertTrue(t["description"])
        json.dumps(tools)  # serializable

    def test_notification_gets_no_reply_and_unknowns_error(self):
        self.assertIsNone(server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(rpc("nope")["error"]["code"], -32601)
        self.assertEqual(rpc("tools/call", {"name": "nope"})["error"]["code"], -32602)
        self.assertEqual(rpc("ping")["result"], {})

    def test_allowlist_limits_listed_and_callable_tools(self):
        os.environ["MESH_TOOLS_ENABLE"] = "math_solve, stats_test"
        try:
            names = {t["name"] for t in rpc("tools/list")["result"]["tools"]}
            blocked = rpc("tools/call", {"name": "literature_search", "arguments": {"query": "x"}})
        finally:
            os.environ.pop("MESH_TOOLS_ENABLE")
        self.assertEqual(names, {"math_solve", "stats_test"})
        self.assertEqual(blocked["error"]["code"], -32602)

    def test_invalid_input_is_tool_error_not_crash(self):
        r = rpc("tools/call", {"name": "stats_test", "arguments": {"kind": "welch", "data_json": "{bad"}})["result"]
        self.assertTrue(r["isError"])
        self.assertIn("invalid input", r["content"][0]["text"])

    def test_every_tool_output_is_capped(self):
        orig = server.BY_NAME["math_solve"]["fn"]
        server.BY_NAME["math_solve"]["fn"] = lambda a: "x" * 50000
        try:
            r = rpc("tools/call", {"name": "math_solve", "arguments": {"code": "1"}})["result"]
        finally:
            server.BY_NAME["math_solve"]["fn"] = orig
        self.assertLessEqual(len(r["content"][0]["text"]), MAX_OUT)
        self.assertIn("[truncated", r["content"][0]["text"])

    def test_stdio_loop(self):
        import subprocess
        root = Path(__file__).resolve().parents[3]
        lines = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                 {"jsonrpc": "2.0", "method": "notifications/initialized"},
                 {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        p = subprocess.run([sys.executable, "-m", "tools.mesh_mcp.server"], cwd=root, capture_output=True,
                           input="\n".join(json.dumps(x) for x in lines) + "\nnot json\n", text=True,
                           encoding="utf-8", timeout=60)
        out = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual([o["id"] for o in out], [1, 2])
        self.assertEqual(len(out[1]["result"]["tools"]), len(EXPECTED))


class CommonTest(unittest.TestCase):
    def test_cap_and_as_json(self):
        self.assertEqual(cap("abc\x00"), "abc")
        self.assertEqual(as_json('{"a": 1}', "x"), {"a": 1})
        self.assertEqual(as_json([1], "x"), [1])
        with self.assertRaises(ValueError):
            as_json("", "x")


class ReportTest(unittest.TestCase):
    def test_imrad_report_saved_with_audit_table(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["MESH_REPORTS_DIR"] = d
            try:
                out = report.report_imrad(
                    "Does X cause Y?",
                    {"introduction": "Why.", "methods": "How.", "results": "Effect 2.0 [1.8, 2.1].",
                     "discussion": "Means.", "abstract": "Short.", "references": ["Ref A", "Ref B"],
                     "appendix_data": "extra"},
                    [{"ts": "t1", "agent": "a", "tool": "causal_analysis", "action": "estimate | ATE",
                      "result": "2.0\nok"}, "free text step"])
            finally:
                os.environ.pop("MESH_REPORTS_DIR")
            files = list(Path(d).glob("*-does-x-cause-y.md"))
            self.assertEqual(len(files), 1)
            md = files[0].read_text(encoding="utf-8")
        self.assertIn("saved ", out)
        heads = [ln for ln in md.splitlines() if ln.startswith("## ")]
        self.assertEqual(heads, ["## Abstract", "## Introduction", "## Methods", "## Results", "## Discussion",
                                 "## Appendix Data", "## References", "## Appendix A. Audit trail"])
        self.assertIn("| 1 | t1 | a | causal_analysis | estimate \\| ATE | 2.0 ok |", md)
        self.assertIn("| 2 |  |  |  | free text step |  |", md)
        self.assertIn("1. Ref A", md)
        self.assertNotIn("Missing IMRaD", md)

    def test_missing_sections_and_audit_flagged(self):
        md = report.build("T", {"introduction": "x"}, [])
        self.assertIn("Missing IMRaD sections: Methods, Results, Discussion", md)
        self.assertIn("No audit trail supplied", md)
        with self.assertRaises(ValueError):
            report.build("T", ["not", "a", "dict"], [])


def _train_modules():
    return {m: sys.modules[m] for m in list(sys.modules) if m == "train" or m.startswith("train.")}


class WorkspaceTest(unittest.TestCase):
    """Swaps fake backends in and out; restores sys.path and the real train.* modules for later suites."""

    def setUp(self):
        self._path, self._mods, self._env = list(sys.path), _train_modules(), os.environ.get("MESH_WORKSPACE_ROOT")
        workspace_tools._mod = None
        for m in self._mods:
            sys.modules.pop(m)

    def tearDown(self):
        workspace_tools._mod = None
        for m in _train_modules():
            sys.modules.pop(m)
        sys.modules.update(self._mods)
        sys.path[:] = self._path
        if self._env is None:
            os.environ.pop("MESH_WORKSPACE_ROOT", None)
        else:
            os.environ["MESH_WORKSPACE_ROOT"] = self._env

    def test_not_available_message(self):
        # simulate an absent backend explicitly: the real one may be importable (e.g. on the integration branch)
        def no_backend(name, *a, **k):
            raise ModuleNotFoundError("No module named %r" % name)
        with mock.patch.object(workspace_tools.importlib, "import_module", no_backend):
            for text in (workspace_tools.workspace_read(), workspace_tools.memory_search("q"),
                         workspace_tools.analogies("q"), workspace_tools.workspace_post("c"),
                         workspace_tools.workspace_stats()):
                self.assertTrue(text.startswith("workspace not available"), text)
        self.assertIsNone(workspace_tools._mod)

    def test_guard_refuses_live_database(self):
        from tools.mesh_mcp import tests as T

        class Live:
            DB = T.LIVE_DB
        workspace_tools._mod = Live()
        # match by message: under pytest this package may also be imported as mesh_mcp.tests (a second class)
        with self.assertRaisesRegex(AssertionError, "refusing to run"):
            workspace_tools.workspace_read()

        class Elsewhere:
            DB = Path(tempfile.gettempdir()) / "other" / "workspace.db"
        workspace_tools._mod = Elsewhere()
        with self.assertRaisesRegex(AssertionError, "refusing to run"):
            workspace_tools.workspace_stats()
    def test_wraps_api_and_defaults_task_id(self):
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d, "train", "memory")
            pkg.mkdir(parents=True)
            Path(d, "train", "__init__.py").write_text("")
            Path(pkg, "__init__.py").write_text("")
            Path(pkg, "workspace.py").write_text(textwrap.dedent("""
                CALLS = []
                def post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None):
                    CALLS.append(("post", task_id, agent, kind, content, confidence, verified, sources, field))
                    return None if "SECRET" in content else 7
                def read(task_id, query=None, k=8):
                    CALLS.append(("read", task_id, query, k))
                    return [{"id": 7, "kind": "fact", "content": "c", "salience": 0.9}]
                def search(text, field=None, k=3):
                    return []
                def analogies(text, k=3, field=None):
                    CALLS.append(("analogies", text, k, field))
                    return [{"id": "card:1", "field": "physics", "text": "heat equation"}]
                def observe(ids, passed):
                    CALLS.append(("observe", ids, passed))
                def stats():
                    return {"entries": 1, "tasks": 1}
            """))
            os.environ["MESH_WORKSPACE_ROOT"] = d
            os.environ["CLAUDE_SESSION_ID"] = "sess-9"
            try:
                self.assertEqual(workspace_tools.workspace_post("c", "fact", agent="me", confidence=0.9, verified=True,
                                                                sources="u", field="math"),
                                 "posted entry 7 to workspace task sess-9")
                self.assertIn("not stored", workspace_tools.workspace_post("SECRET x", "fact"))
                self.assertIn('1. {"id": 7', workspace_tools.workspace_read(query="q", k=2))
                self.assertEqual(workspace_tools.memory_search("q"), "no matching memory")
                self.assertIn("heat equation", workspace_tools.analogies("q", 2, "math"))
                self.assertIn("2 item(s)", workspace_tools.workspace_observe(["7", "card:1"], True))
                self.assertIn('"entries": 1', workspace_tools.workspace_stats())
                with self.assertRaises(ValueError):
                    workspace_tools.workspace_observe([], True)
                calls = sys.modules["train.memory.workspace"].CALLS
            finally:
                os.environ.pop("MESH_WORKSPACE_ROOT")
                os.environ.pop("CLAUDE_SESSION_ID")
                while d in sys.path:
                    sys.path.remove(d)
        # verified=True from the agent is never forwarded; the claim travels as a content tag
        self.assertEqual(calls[0], ("post", "sess-9", "me", "fact", workspace_tools.CLAIM_TAG + "c", 0.9, False, ["u"],
                                    "math"))
        self.assertEqual(calls[2], ("read", "sess-9", "q", 2))
        self.assertEqual(calls[3], ("analogies", "q", 2, "math"))
        self.assertEqual(calls[4], ("observe", [7, "card:1"], True))

    def test_verified_claim_goes_to_claimed_verified_when_backend_supports_it(self):
        calls = []

        class Backend:
            @staticmethod
            def post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None,
                     claimed_verified=False, **extra):
                assert not extra, extra  # verified_by / approval_sig must never be forwarded
                calls.append({"content": content, "verified": verified, "claimed_verified": claimed_verified})
                return 3

            @staticmethod
            def read(task_id, query=None, k=8):
                return [{"id": 3, "verified": True, "verified_by": "human:m", "approval_sig": "SIG123"}]

        workspace_tools._mod = Backend()
        workspace_tools.workspace_post("x", "result", "t", verified=True)
        workspace_tools.workspace_post("y", "result", "t", verified=False)
        self.assertEqual(calls, [{"content": "x", "verified": False, "claimed_verified": True},
                                 {"content": "y", "verified": False, "claimed_verified": False}])
        out = workspace_tools.workspace_read("t")
        self.assertNotIn("SIG123", out)
        self.assertNotIn("approval_sig", out)
        self.assertIn('"verified_by": "human:m"', out)
        names = {t["name"] for t in server.TOOLS}
        self.assertFalse({"approve", "workspace_approve"} & names)
        self.assertNotIn("verified_by", json.dumps([t["inputSchema"] for t in server.TOOLS]))


@unittest.skipUnless(os.environ.get("MESH_WORKSPACE_IT"), "set MESH_WORKSPACE_IT=<repo with train/memory/workspace.py>")
class WorkspaceIntegrationTest(unittest.TestCase):
    """Against the real module, with a throwaway database and key."""

    def setUp(self):
        self._path, self._mods = list(sys.path), _train_modules()

    def tearDown(self):
        for m in _train_modules():
            sys.modules.pop(m)
        sys.modules.update(self._mods)
        sys.path[:] = self._path

    def test_round_trip(self):
        workspace_tools._mod = None
        for m in [m for m in sys.modules if m == "train" or m.startswith("train.")]:
            sys.modules.pop(m)
        from tools.mesh_mcp import tests as T
        with tempfile.TemporaryDirectory(dir=T.TMP) as d:  # inside the guarded test dir
            env = {"MESH_WORKSPACE_ROOT": os.environ["MESH_WORKSPACE_IT"], "MESH_WORKSPACE_DB": str(Path(d, "w.db")),
                   "MESH_WORKSPACE_KEY": str(Path(d, "w.key"))}
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                out = workspace_tools.workspace_post("Gaussian integral equals sqrt(pi)", "result", "it-task", "t",
                                                     0.9, True, ["math_solve"], "math")
                self.assertRegex(out, r"^posted entry \d+ to workspace task it-task$")
                out = workspace_tools.workspace_read("it-task", "gaussian")
                self.assertIn("sqrt(pi)", out)
                self.assertIn('"verified": false', out)  # the agent's verified=True was not trusted
                self.assertTrue('"claimed_verified": true' in out or "agent claims verified" in out, out)
                self.assertNotIn("approval_sig", out)
                self.assertIn('"entries": 1', workspace_tools.workspace_stats())
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                workspace_tools._mod = None
                for m in [m for m in sys.modules if m == "train" or m.startswith("train.")]:
                    sys.modules.pop(m)

if __name__ == "__main__":
    unittest.main()
