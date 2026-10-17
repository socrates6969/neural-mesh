"""Regression tests for two leak paths: a secret in prose ("the password is X") and text written after reading a sensitive file."""
import os
import unittest
from pathlib import Path

from ingest.pipeline import run
from ingest.scrub import scrub
from ingest.tests.test_pipeline import Env, asst, tool_result, user

PROSE = "CanaryProse-7xQ2"
ECHO = "hunter2-CANARY-echo"


class LeakFixTests(unittest.TestCase):
    def test_prose_password_is_redacted_and_flagged(self):
        r = scrub(f"btw the staging db password is {PROSE} - use it")
        self.assertNotIn(PROSE, r.text)
        self.assertTrue(r.secret)
        self.assertFalse(scrub("the password is required and the token is expired").secret)   # prose words are not secrets

    def test_text_after_sensitive_read_never_reaches_any_output(self):
        os.environ["INGEST_EMBED_EXE"] = "Z:/missing/embed_texts.exe"
        env = Env()
        try:
            env.write("proj/s1.jsonl", [
                user(f"Check the config please; the staging password is {PROSE}", 0),
                asst([{"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": "C:/w/.env"}}], 1, "m1"),
                tool_result("r1", f"DB_PASSWORD={ECHO}", 2),
                asst([{"type": "text", "text": f"Found it: {ECHO} is the staging value, nothing else to change."}], 3, "m2"),
                user("thanks", 10),
            ])
            run(env.cfg)
            for p in Path(env.data).rglob("*"):
                if p.is_file():
                    blob = p.read_bytes().decode("utf-8", "replace")
                    self.assertNotIn(ECHO, blob, p.name)
                    self.assertNotIn(PROSE, blob, p.name)
        finally:
            os.environ.pop("INGEST_EMBED_EXE", None)
            env.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
