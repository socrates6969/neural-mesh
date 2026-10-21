import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ingest import catalog


def _w(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        c = self.home / ".claude"
        _w(c / "agents" / "core" / "coder.md", "---\nname: coder\ndescription: \"Writes code\"\n---\nbody")
        _w(c / "agents" / "nodesc.md", "---\nname: nodesc\n---\nbody")
        plug = self.home / "p1"
        _w(plug / "agents" / "seo-auditor" / "AGENT.md", "---\ndescription: Runs an SEO audit\n---\n")
        _w(plug / "agents" / "named.md", "---\nname: lawyer\ndescription: >\n  Reviews\n  contracts\n---\n")
        _w(plug / "agents" / "bare.md", "---\ntools: Read\n---\n")
        off = self.home / "p2"
        _w(off / "agents" / "x.md", "---\nname: x\ndescription: disabled one\n---\n")
        _w(c / "settings.json", "{}")
        self.plugins = [{"id": "seo@synced", "installPath": str(plug), "enabled": True},
                        {"id": "off@m", "installPath": str(off), "enabled": False}]

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        with mock.patch.object(catalog, "_plugin_list", return_value=self.plugins):
            return {r["name"]: r for r in catalog.build(self.home)}

    def test_sources_names_and_enabled(self):
        rows = self.build()
        self.assertIn("general-purpose", rows)
        self.assertEqual(rows["coder"]["source"], "user")
        self.assertNotIn("nodesc", rows)                                  # user agent without description is skipped
        self.assertIn("seo:seo-auditor:AGENT", rows)                       # path-based name, like Claude Code
        self.assertEqual(rows["seo:lawyer"]["description"], "Reviews contracts")   # folded block scalar
        self.assertEqual(rows["seo:bare"]["description"], "Agent from seo plugin")
        self.assertFalse(rows["off:x"]["enabled"])
        self.assertTrue(rows["seo:lawyer"]["enabled"])

    def test_no_agent_bodies_copied(self):
        rows = self.build()
        self.assertNotIn("body", json.dumps(rows))

    def test_write_atomic(self):
        out = self.home / "out" / "cat.json"
        res = catalog.write(out, list(self.build().values()))
        self.assertEqual(res["total"], len(json.loads(out.read_text(encoding="utf-8"))["agents"]))
        self.assertFalse(out.with_suffix(".tmp").exists())


class RosterTest(unittest.TestCase):
    def test_roster_shape(self):
        r = json.loads((Path(__file__).resolve().parents[2] / "catalog" / "roster.json").read_text(encoding="utf-8"))
        for field, agents in r["fields"].items():
            self.assertTrue(1 <= len(agents) <= 3, field)
            self.assertEqual(len(agents), len(set(agents)), field)


if __name__ == "__main__":
    unittest.main()
