"""Scrubber tests with SYNTHETIC fake secrets only (built at runtime; none is a real credential)."""
import random
import string
import unittest
from collections import Counter

from ingest.scrub import SECRET_CATEGORIES, looks_high_entropy, scrub, touches_sensitive_path


def fake(n, alphabet=string.ascii_letters + string.digits, seed=7):
    r = random.Random(seed)
    return "".join(r.choice(alphabet) for _ in range(n))


class SecretRedaction(unittest.TestCase):
    def check(self, text, cat):
        r = scrub(text)
        self.assertIn(cat, r.hits, f"{cat} not detected")
        self.assertTrue(r.secret)
        return r

    def test_provider_keys(self):
        for key in ["sk-ant-api03-" + fake(40), "sk-" + fake(48), "AKIA" + fake(16, string.ascii_uppercase + string.digits),
                    "ghp_" + fake(36), "github_pat_" + fake(40, string.ascii_letters + string.digits + "_"), "xoxb-" + fake(30),
                    "AIza" + fake(35), "npm_" + fake(36), "hf_" + fake(34), "sk_live_" + fake(24), "glpat-" + fake(20)]:
            r = self.check(f"use this key {key} please", "api_key")
            self.assertNotIn(key, r.text)

    def test_jwt(self):
        jwt = "eyJ" + fake(20, seed=1) + "." + fake(30, seed=2) + "." + fake(25, seed=3)
        r = self.check(f"Authorization header {jwt}", "jwt")
        self.assertNotIn(jwt, r.text)

    def test_private_key_block(self):
        block = "-----BEGIN RSA PRIVATE KEY-----\n" + fake(64) + "\n" + fake(64, seed=9) + "\n-----END RSA PRIVATE KEY-----"
        r = self.check("key:\n" + block + "\ndone", "private_key")
        self.assertNotIn(fake(64), r.text)
        self.assertIn("done", r.text)

    def test_connection_string(self):
        pw = fake(14)
        r = self.check(f"DATABASE_URL=postgresql://app_user:{pw}@db.internal:5432/main", "conn_string")
        self.assertNotIn(pw, r.text)

    def test_password_assignment(self):
        pw = "Tr0ub4dor" + fake(6)
        r = self.check(f'password = "{pw}"', "password")
        self.assertNotIn(pw, r.text)
        r = self.check(f"client_secret: {fake(20)}", "password")

    def test_bearer(self):
        tok = fake(32)
        r = self.check(f"curl -H 'Authorization: Bearer {tok}'", "bearer")
        self.assertNotIn(tok, r.text)

    def test_high_entropy_token(self):
        tok = fake(40, seed=11)
        r = self.check(f"blob {tok} end", "high_entropy")
        self.assertNotIn(tok, r.text)

    def test_card_number_luhn(self):
        r = self.check("card 4111 1111 1111 1111 exp", "card_number")  # the standard public test number
        self.assertNotIn("4111 1111 1111 1111", r.text)


class PiiRedaction(unittest.TestCase):
    def test_email_redacted_not_secret(self):
        r = scrub("mail someone.fake@example.org about it")
        self.assertEqual(r.hits["email"], 1)
        self.assertNotIn("someone.fake@example.org", r.text)
        self.assertFalse(r.secret)

    def test_phone(self):
        r = scrub("call +47 912 34 567 now")
        self.assertIn("phone", r.hits)


class NoFalsePositives(unittest.TestCase):
    def test_ordinary_code_and_hashes(self):
        texts = [
            "commit 5828944 docs: assert ownership; sha 3f2a9c1e5b7d4a6f8e0c2b4d6f8a0c2e4b6d8f0a",
            "session 088e806e-524a-40de-95bf-b8a313fa3fc6 finished",
            "const password: string = input.password; // typed field",
            "password: z.string().min(12)",
            "Run cargo test --release --lib and npm run typecheck",
            "token budget is 300 and api_key: process.env.API_KEY",
            "C:/Users/mariu/neural-mesh/src/encoder.rs:829 encode_batch",
            "tests: 403 passed, 0 failed",
        ]
        for t in texts:
            r = scrub(t)
            self.assertFalse(r.secret, f"false positive on: {t} -> {dict(r.hits)}")

    def test_entropy_rules(self):
        self.assertFalse(looks_high_entropy("a" * 40))
        self.assertFalse(looks_high_entropy("3f2a9c1e5b7d4a6f8e0c2b4d6f8a0c2e4b6d8f0a"))  # hex digest
        self.assertTrue(looks_high_entropy(fake(40, seed=5)))


class SensitivePaths(unittest.TestCase):
    def test_touches(self):
        yes = [".env", "api/.env.local", "cat .env.example", "C:/repo/secrets/db.txt", "server.key", "certs/tls.pem",
               "~/.ssh/id_ed25519", "~/.aws/credentials", "credentials.json", ".npmrc", "token.json", "~/.docker/config.json"]
        no = ["src/environment.ts", "README.md", "src/secretary.ts", "docs/secrets.md", "keyboard.rs", "api/src/auth/credentials.ts",
              "npm run test", "src/env.rs"]
        for p in yes:
            self.assertTrue(touches_sensitive_path(p), p)
        for p in no:
            self.assertFalse(touches_sensitive_path(p), p)

    def test_mention_redacted_but_kept(self):
        r = scrub("never read .env files")
        self.assertIn("sensitive_path_mention", r.hits)
        self.assertNotIn(".env", r.text)
        self.assertFalse(r.secret)


class Counting(unittest.TestCase):
    def test_counts_only(self):
        c = Counter()
        scrub("a@b.co and c@d.co", c)
        scrub("e@f.co", c)
        self.assertEqual(c["email"], 3)
        self.assertTrue(all(isinstance(v, int) for v in c.values()))
        self.assertTrue(SECRET_CATEGORIES.isdisjoint({"email", "phone", "sensitive_path_mention"}))


if __name__ == "__main__":
    unittest.main()
