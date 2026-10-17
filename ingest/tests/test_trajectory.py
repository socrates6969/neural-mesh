"""gap B/C/D/G: content-free call records, execution evidence, per-call taint, any-depth discovery, and the canary leak test."""
import json
import os
import unittest
from pathlib import Path

from ingest import trajectory as tj
from ingest.pipeline import run, verdict_of
from ingest.reader import classify_rel
from ingest.tests.test_pipeline import Env, asst, tool_result, user

CANARIES = {
    "api_key_in_command": "sk-ant-CANARYapikey0000000000000000",
    "plain_password_from_env_file": "hunter2-CANARY-plainpw",
    "file_content_in_result": "CANARYfilecontent-7b1",
    "grep_pattern": "CANARYgreppattern",
    "edit_new_string": "CANARYeditbody-99",
    "kv_password_in_text": "CanaryPw-Zq81mXk2",
    "file_name": "CANARYfilename",
    "prompt_password": "CANARYpromptpw-4d2",
}


def bash(tid, cmd):
    return {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": cmd, "description": "run"}}


class TrajectoryUnitTests(unittest.TestCase):
    def test_shell_verbs_ignore_quoted_text_and_heredocs(self):
        self.assertEqual(tj.shell_verbs('cd /x && cargo test 2>&1 | tail -5'), ["cargo test", "tail"])
        self.assertEqual(tj.shell_verbs('grep -rE "spawn|secretword|x" src | head'), ["grep", "head"])
        self.assertEqual(tj.shell_verbs('"C:/py/python.exe" -m pytest -q'), ["python -m pytest"])
        self.assertEqual(tj.shell_verbs("python - <<EOF\nimport os\nos.listdir('.')\nEOF"), ["python"])
        self.assertEqual(tj.check_kind(tj.shell_verbs("npm run build && npx vitest run")), "test")
        self.assertEqual(tj.check_kind(tj.shell_verbs("FOO=1 cargo clippy --all")), "build")
        self.assertIsNone(tj.check_kind(tj.shell_verbs("git status")))

    def test_call_taint_classes(self):
        b = lambda c: tj.call_taint("Bash", {"command": c}, c, [])  # noqa: E731
        self.assertEqual(b("cat .env"), "read")
        self.assertEqual(b("git diff .env"), "read")
        self.assertEqual(b("ls -la .env"), "mention")
        self.assertEqual(b("git status .env"), "mention")
        self.assertIsNone(b("grep KEY .env.example"))                  # template allowlisted
        self.assertIsNone(b('python -c "print(x.key)"'))                # attribute access, not a .key file
        self.assertEqual(b("cat config/server.key"), "read")
        self.assertEqual(tj.call_taint("Glob", {"pattern": "**/.env"}, "**/.env", []), "mention")
        self.assertEqual(tj.call_taint("Write", {"file_path": "a/.env"}, "a/.env", []), "write")
        self.assertEqual(tj.call_taint("Read", {"file_path": "a/.env"}, "a/.env", []), "read")

    def test_tainted_ignores_failed_read_and_mentions(self):
        self.assertFalse(tj.tainted([{"n": "Read", "tt": "read", "ok": False}, {"n": "Bash", "tt": "mention", "ok": True}]))
        self.assertTrue(tj.tainted([{"n": "Read", "tt": "read", "ok": True}]))
        self.assertTrue(tj.tainted([{"n": "Bash", "tt": "read", "ok": False}]))     # a failing shell command may still print the file
        self.assertTrue(tj.tainted([{"n": "Write", "tt": "write", "ok": False}]))

    def test_result_summary_counts_only(self):
        r = tj.result_summary("Exit code 101\ntest result: FAILED. 3 passed; 1 failed", True)
        self.assertEqual(r, {"ok": False, "ex": 101, "p": 3, "f": 1})
        self.assertTrue(tj.result_summary("FAILED tests/test_x.py::test_y - assert 1 == 2", False).get("ce"))
        self.assertTrue(all(isinstance(v, int | bool) for v in tj.result_summary("2 passed, 1 failed in 0.3s", False).values()))

    def test_result_summary_v3_cases(self):
        self.assertTrue(tj.result_summary("Command running in background with ID: b1. Output is being written to: x", False).get("bg"))
        self.assertIsNone(tj.check_pass({"c": "test", "ok": True, "bg": True}))
        r = tj.result_summary("\u2139 tests 278\n\u2139 pass 268\n\u2139 fail 7\n", False)
        self.assertEqual((r["p"], r["f"]), (268, 7))
        self.assertTrue(tj.result_summary("Version 1.52.0\nexit 124\n", False).get("ce"))
        self.assertFalse(tj.result_summary("766 passed in 12.71s\nexit=0\n", False).get("ce"))
        self.assertTrue(tj.shell_writes_code("cd x && cat > src/a.py <<'EOF'\nprint(1)\nEOF"))
        self.assertTrue(tj.shell_writes_code("sed -i 's/a/b/' src/lib.rs && cargo test"))
        self.assertFalse(tj.shell_writes_code("cargo test 2>&1 | tail -5 > /tmp/out.log"))
        self.assertFalse(tj.shell_writes_code("grep -n 'x > y.py' src"))

    def test_pass_needs_positive_evidence_and_script_writes_are_edits(self):
        silent = dict(tj.result_summary("(Bash completed with no output)", False), c="test")
        self.assertIsNone(tj.check_pass(silent))                                     # a silent check proves nothing
        self.assertTrue(tj.check_pass(dict(tj.result_summary("Ran 12 tests in 0.1s\n\nOK\n", False), c="test")))
        self.assertTrue(tj.check_pass(dict(tj.result_summary("    Finished `release` profile [optimized] target(s) in 3s", False), c="build")))
        self.assertGreaterEqual(tj.write_pos("python - <<'EOF'\np='src/x.rs'\nopen(p,'w').write(s)\nEOF"), 0)
        self.assertEqual(tj.write_pos("python - <<'EOF'\nimport json\nprint(json.load(open('r.json')))\nEOF"), -1)
        cmd = "cargo test 2>&1 | tail -3; cat > run_all.sh <<'EOF'\necho hi\nEOF"
        self.assertGreater(tj.write_pos(cmd), tj.check_pos(cmd))                    # the write comes after the test in one call

    def test_evidence(self):
        edit = {"k": "edit", "x": ".rs", "ok": True}
        doc = {"k": "edit", "x": ".md", "ok": True}
        ok_test = {"k": "shell", "c": "test", "ok": True, "p": 4, "f": 0}
        bad_test = {"k": "shell", "c": "test", "ok": True, "p": 3, "f": 1}
        self.assertEqual(tj.evidence([edit, bad_test, edit, ok_test, doc])["verified"], True)     # doc edit after the check is fine
        self.assertEqual(tj.evidence([edit, ok_test, edit])["verified"], None)                    # code edited after the last check
        self.assertEqual(tj.evidence([edit, ok_test, bad_test])["verified"], False)
        self.assertEqual(tj.evidence([{"k": "read", "ok": True}])["why"], "no_edit_no_check")
        both = {"k": "shell", "c": "test", "we": True, "ok": True, "p": 5, "f": 0}          # sed -i ... && pytest in ONE call
        self.assertEqual(tj.evidence([edit, both])["verified"], True)
        shell_edit = {"k": "shell", "we": True, "ok": True}
        self.assertIsNone(tj.evidence([edit, ok_test, shell_edit])["verified"])          # code written by the shell after the check

    def test_verdicts(self):
        self.assertTrue(verdict_of("Summary...\n**Verdict: PASS** - all criteria met"))
        self.assertFalse(verdict_of("## Result: REJECTED\nD1 open"))
        self.assertIsNone(verdict_of("tests pass, looks fine"))

    def test_any_depth_discovery(self):
        self.assertEqual(classify_rel("p/s.jsonl"), "main")
        self.assertEqual(classify_rel("p/s/subagents/agent-a.jsonl"), "agent")
        self.assertEqual(classify_rel("p/s/subagents/team/x/agent-b.jsonl"), "agent")
        self.assertIsNone(classify_rel("p/s/tool-results/out.jsonl"))
        self.assertIsNone(classify_rel("top.jsonl"))


class CanaryLeakTest(unittest.TestCase):
    """Seed fake secrets everywhere a transcript can carry them, run the whole pipeline, grep every output byte."""

    def setUp(self):
        os.environ["INGEST_EMBED_EXE"] = "Z:/missing/embed_texts.exe"
        self.env = Env()
        c = CANARIES
        sess = "sess-canary"
        main = [
            user("Fix the parser in src/parse.rs and make the tests pass please", 0),
            asst([{"type": "tool_use", "id": "tA", "name": "Agent", "input": {"subagent_type": "coder", "description": "fix parser",
                  "prompt": "Fix src/parse.rs so that cargo test passes. Do not change the public API."}}], 1, "m1"),
            tool_result("tA", f"Done, cargo test: 12 passed. Staging uses {c['plain_password_from_env_file']}.", 30),
            asst([{"type": "text", "text": "Fixed: the parser now handles empty input; all 12 tests pass after the change. "
                                           f"For staging log in with {c['plain_password_from_env_file']}."}], 31, "m2"),
            user(f"also, the staging db password is {c['prompt_password']} - check it works", 40),
            asst([{"type": "text", "text": "I checked the connection settings; nothing else needed to change in the parser code today."}], 41, "m3"),
            asst([{"type": "tool_use", "id": "tB", "name": "Agent", "input": {"subagent_type": "coder", "description": "tok test",
                  "prompt": "Add a unit test for the tokenizer in src/tok.rs and run the suite."}}], 42, "m4"),
            tool_result("tB", "Added the test; 13 passed.", 50),
        ]
        self.env.write(f"proj/{sess}.jsonl", main)
        sub = [
            user("Fix src/parse.rs so that cargo test passes. Do not change the public API.", 2, True),
            asst([{"type": "text", "text": "Let me look at the environment first."},
                  bash("s1", f"export ANTHROPIC_API_KEY={c['api_key_in_command']} && cargo test 2>&1 | tail -3")], 3, "s1"),
            tool_result("s1", "Exit code 101\ntest result: FAILED. 11 passed; 1 failed", 4),
            asst([{"type": "tool_use", "id": "s2", "name": "Read", "input": {"file_path": "C:/work/demo-proj/.env"}}], 5, "s2"),
            tool_result("s2", f"DB_PASSWORD={c['plain_password_from_env_file']}\n", 6),
            asst([bash("s3", f'grep -rn "{c["grep_pattern"]}|parse_empty" src | head -5')], 7, "s3"),
            tool_result("s3", f"src/parse.rs:10: // {c['file_content_in_result']}", 8),
            asst([{"type": "text", "text": f"The config has password = {c['kv_password_in_text']} which is fine. Now the fix."},
                  {"type": "tool_use", "id": "s4", "name": "Edit", "input": {"file_path": f"C:/work/demo-proj/src/{c['file_name']}.rs",
                   "old_string": "fn a()", "new_string": c["edit_new_string"]}}], 9, "s4"),
            tool_result("s4", "The file has been updated.", 10),
            asst([bash("s5", "cargo test 2>&1 | tail -3")], 11, "s5"),
            tool_result("s5", "test result: ok. 12 passed; 0 failed", 12),
            asst([{"type": "text", "text": f"Fixed the empty-input case. Note: {c['plain_password_from_env_file']} is what staging uses. "
                                           "cargo test: 12 passed, 0 failed."}], 13, "s6"),
        ]
        self.env.write(f"proj/{sess}/subagents/nested/agent-ccc.jsonl", sub)       # nested deeper than the old fixed layout
        (self.env.projects / f"proj/{sess}/subagents/nested/agent-ccc.meta.json").write_text(
            json.dumps({"agentType": "coder", "toolUseId": "tA", "spawnDepth": 1}))
        clean = [   # a clean second agent (no sensitive access) whose rows SHOULD be produced
            user("Add a unit test for the tokenizer in src/tok.rs and run the suite.", 43, True),
            asst([{"type": "text", "text": "Adding the test."},
                  {"type": "tool_use", "id": "k1", "name": "Edit", "input": {"file_path": "C:/work/demo-proj/src/tok.rs", "old_string": "x",
                   "new_string": "y"}}], 44, "k1"),
            tool_result("k1", "ok", 45),
            asst([{"type": "text", "text": "Running the tests now."}, bash("k2", "cargo test")], 46, "k2"),
            tool_result("k2", "test result: ok. 13 passed; 0 failed", 47),
            asst([{"type": "text", "text": "Added test_tokenizer_handles_unicode in src/tok.rs; cargo test: 13 passed, 0 failed."}], 48, "k3"),
        ]
        self.env.write(f"proj/{sess}/subagents/agent-ddd.jsonl", clean)
        (self.env.projects / f"proj/{sess}/subagents/agent-ddd.meta.json").write_text(
            json.dumps({"agentType": "coder", "toolUseId": "tB", "spawnDepth": 1}))
        self.env.write(f"proj/{sess}/tool-results/not-a-transcript.jsonl", [{"x": 1}])

    def tearDown(self):
        os.environ.pop("INGEST_EMBED_EXE", None)
        self.env.tmp.cleanup()

    def test_no_canary_reaches_any_output_and_clean_rows_are_produced(self):
        summary = run(self.env.cfg)
        self.assertEqual(summary["files"], 3)
        self.assertEqual(summary["files_skipped"], 1)
        leaks = {}
        for p in Path(self.env.data).rglob("*"):
            if p.is_file():
                blob = p.read_bytes().decode("utf-8", "replace").lower()
                for name, secret in CANARIES.items():
                    if secret.lower() in blob:
                        leaks.setdefault(name, []).append(p.name)
        self.assertEqual(leaks, {}, f"canary leaks: {leaks}")
        traj = [json.loads(x) for x in open(self.env.data / "trajectories.jsonl", encoding="utf-8")]
        self.assertTrue(any(t["tainted"] for t in traj))                        # the .env read taints its agent
        clean = {t["id"] for t in traj if not t["tainted"] and t["verified"] is True}
        self.assertTrue(clean)                                                   # the clean agent is verified by its passing cargo test
        steps = [json.loads(x) for x in open(self.env.data / "tooluse_sft.jsonl", encoding="utf-8")]
        self.assertTrue(steps)
        self.assertTrue(all(s["traj"] in clean for s in steps if s["verified"] is True))
        sft = [json.loads(x) for x in open(self.env.data / "sft.jsonl", encoding="utf-8")]
        self.assertTrue(any(r["verified"] is True and "tokenizer" in r["instruction"] for r in sft))


if __name__ == "__main__":
    unittest.main()