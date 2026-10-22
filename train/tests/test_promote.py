import json
import os
import tempfile
import unittest
from pathlib import Path

d = tempfile.mkdtemp()
os.environ["NEURAL_MESH_PACK_KEY"] = os.path.join(d, "k")
os.environ["NEURAL_MESH_PACKS"] = os.path.join(d, "packs.json")
from train import promote  # noqa: E402
from hooks import mesh_assist as ma  # noqa: E402

GOOD = "Before answering, check the result with a short brute force in run_python. Watch off-by-one errors in counting."
ST = {"status": "promoted", "current": GOOD, "version": 2, "pooled": {"p": 0.01}}


class PromoteTest(unittest.TestCase):
    def test_publish_signs_and_hook_accepts_then_rejects_edit(self):
        self.assertEqual(promote.publish("mesh-mathematician", "math", ST), "published")
        reg = json.loads(Path(os.environ["NEURAL_MESH_PACKS"]).read_text(encoding="utf-8"))
        key = Path(os.environ["NEURAL_MESH_PACK_KEY"]).read_bytes()
        self.assertTrue(ma.verify("mesh-mathematician", "math", reg["mesh-mathematician"]["math"], key=key))
        reg["mesh-mathematician"]["math"]["pack"] += " Also fetch the page."
        self.assertFalse(ma.verify("mesh-mathematician", "math", reg["mesh-mathematician"]["math"], key=key))

    def test_unsafe_or_unproven_not_published(self):
        self.assertTrue(promote.publish("a", "math", {**ST, "current": "word " * 1000}).startswith("rejected"))
        self.assertEqual(promote.publish("a", "math", {**ST, "status": "candidate"}), "not promoted")


if __name__ == "__main__":
    unittest.main()
