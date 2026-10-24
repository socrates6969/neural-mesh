"""Server protocol, output cap, report formatter and workspace wrapper tests (no network, no Docker)."""
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tools.mesh_mcp import report, server, workspace_tools
from tools.mesh_mcp.common import MAX_OUT, as_json, cap

EXPECTED = {"literature_search", "math_solve", "stats_test", "causal_analysis", "discover_graph",
            "workspace_post", "workspace_read", "memory_search", "analogies", "report_imrad"}


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


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        workspace_tools._mod = None
        for m in [m for m in sys.modules if m == "train" or m.startswith("train.")]:
            sys.modules.pop(m)

    tearDown = setUp

    def test_not_available_message(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["MESH_WORKSPACE_ROOT"] = d
            try:
                for text in (workspace_tools.workspace_read(), workspace_tools.memory_search("q"),
                             workspace_tools.analogies("q"), workspace_tools.workspace_post("c")):
                    self.assertTrue(text.startswith("workspace not available"), text)
            finally:
                os.environ.pop("MESH_WORKSPACE_ROOT")
                sys.path.remove(d)

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
                    return {"id": "w1"}
                def read(task_id, query=None, k=8):
                    CALLS.append(("read", task_id, query, k))
                    return [{"id": "w1", "kind": "finding", "content": "c"}]
                def search(text, field=None, k=3):
                    return []
                def analogies(text, k=3):
                    return ["physics: heat equation"]
            """))
            os.environ["MESH_WORKSPACE_ROOT"] = d
            os.environ["CLAUDE_SESSION_ID"] = "sess-9"
            try:
                self.assertIn('"w1"', workspace_tools.workspace_post("c", "finding", agent="me", confidence=0.9,
                                                                     verified=True, sources=["u"], field="math"))
                self.assertIn("1. {", workspace_tools.workspace_read(query="q", k=2))
                self.assertEqual(workspace_tools.memory_search("q"), "no matching memory")
                self.assertEqual(workspace_tools.analogies("q"), "1. physics: heat equation")
                calls = sys.modules["train.memory.workspace"].CALLS
            finally:
                os.environ.pop("MESH_WORKSPACE_ROOT")
                os.environ.pop("CLAUDE_SESSION_ID")
                sys.path.remove(d)
        self.assertEqual(calls[0], ("post", "sess-9", "me", "finding", "c", 0.9, True, ["u"], "math"))
        self.assertEqual(calls[1], ("read", "sess-9", "q", 2))


if __name__ == "__main__":
    unittest.main()
