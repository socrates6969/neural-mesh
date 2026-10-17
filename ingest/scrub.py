"""Hard scrubber: every piece of transcript text passes through here BEFORE anything is stored.

Policy (LLM-DESIGN.md, pipeline step 2):
* a tool call whose input touches a sensitive path (.env*, secrets/, *.key, *.pem, credentials, tokens, ssh keys, ...) is
  "tainted": its input and its result are never stored, and the turn / agent that made it is excluded from sft and memory;
* regex + entropy redaction of keys, tokens, JWTs, private keys, connection strings, passwords, emails, phone and card numbers;
* "if in doubt, drop": text with any SECRET-class hit is not used for sft or memory at all (``ScrubResult.secret`` is True);
  emails / phones / sensitive-path mentions are only redacted;
* only COUNTS per category are ever logged, never the matched content.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

SCRUB_VERSION = "scrub-v2"   # v2: natural-language secrets ("the password is X"); session secret values (reader.py)

# --- sensitive paths (a tool call touching one of these is dropped) -----------------------------------------------
_SENSITIVE_PATH_PARTS = [
    r"(?<![\w.-])\.env(?:\.[\w.-]+)?(?![\w-])",            # .env, .env.local, .env.example (all of them: if in doubt, drop)
    r"(?:^|[\\/\s\"'`=(])secrets?[\\/]",                      # secrets/ or secret/ directories
    r"[\w.-]+\.(?:pem|key|p12|pfx|jks|keystore|ppk|kdbx|asc|gpg)(?![\w])",
    r"(?:^|[\\/\s\"'`])id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?\b",
    r"[\\/]\.ssh[\\/]",
    r"[\\/]\.aws[\\/]",
    r"[\\/]\.azure[\\/]",
    r"[\\/]\.kube[\\/]config",
    r"[\\/]\.docker[\\/]config\.json",
    r"(?:^|[\\/\s\"'`])\.(?:npmrc|pypirc|netrc|git-credentials|pgpass|htpasswd)\b",
    r"(?:^|[\\/\s\"'`_-])credentials?(?:\.json|\.ya?ml|\.xml|\.txt|\.csv)?(?=$|[\s\"'`)\\/])",
    r"(?:^|[\\/\s\"'`_-])tokens?\.(?:json|txt|ya?ml)\b",
    r"(?:^|[\\/\s\"'`])(?:service[-_]?account|client[-_]?secret)[\w.-]*\.json\b",
    r"(?:^|[\\/\s\"'`])auth\.json\b",
]
SENSITIVE_PATH_RE = re.compile("|".join(_SENSITIVE_PATH_PARTS), re.IGNORECASE)


def touches_sensitive_path(text: str) -> bool:
    """True when a tool call input (as text) references a denylisted path."""
    return bool(text) and SENSITIVE_PATH_RE.search(text) is not None


# --- secret patterns (SECRET class: redacted, and the record is dropped from sft/memory) ---------------------------
_SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key", re.compile(r"-----BEGIN[ A-Z0-9]*PRIVATE KEY-----[\s\S]*?(?:-----END[ A-Z0-9]*PRIVATE KEY-----|$)")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}")),
    ("api_key", re.compile(
        r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{12,}"
        r"|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|glpat-[A-Za-z0-9_-]{16,}"
        r"|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,}|npm_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_-]{30,}"
        r"|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}|ya29\.[A-Za-z0-9_-]{20,}|dop_v1_[a-f0-9]{40,}|shpat_[a-fA-F0-9]{20,})")),
    ("bearer", re.compile(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9_\-.=+/]{16,}")),
    ("conn_string", re.compile(r"(?i)\b[a-z][a-z0-9+.-]{1,20}://[^\s:/@\"'`]+:[^\s@\"'`]+@[^\s\"'`]+")),
    ("password", re.compile(
        r"(?i)(?P<k>\b(?:password|passwd|pwd|pass|secret|api[_-]?key|apikey|access[_-]?key|secret[_-]?key|client[_-]?secret"
        r"|auth[_-]?token|access[_-]?token|refresh[_-]?token|private[_-]?key|session[_-]?secret|signing[_-]?key)\b[\"']?\s*(?:=|:|=>)\s*)"
        r"(?P<q>[\"'`]?)(?P<v>[^\s\"'`,;)}\]]{6,})")),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    # v2: "the db password is X", "api key was X", "token: X" in prose (the key=value form above misses these)
    ("password_phrase", re.compile(
        r"(?i)(?P<k>\b(?:password|passwd|passphrase|pass ?code|pwd|api[ _-]?key|secret(?:[ _-]?key)?|access[ _-]?token|auth[ _-]?token"
        r"|bearer token|token|credentials?)\s+(?:is|was|=|:|set to|of|reads|equals)\s+)(?P<q>[\"'`]?)(?P<v>[^\s\"'`,;)]{6,})")),
]
# --- PII patterns (redacted, record kept) ------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<![\w+])\+\d{1,3}[\s-]?(?:\d[\s-]?){7,12}\d\b")),
]
_HIGH_ENTROPY_CAND = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{24,}(?![A-Za-z0-9+/=_-])")
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_HEX = re.compile(r"^[0-9a-fA-F]+$")

# values that are obviously not secrets in a "password = <v>" style match
_PLACEHOLDER = re.compile(
    r"(?i)^(?:<[^>]*>|\$\{?[A-Za-z_][\w.]*\}?|process\.env\.[\w.]+|os\.environ.*|env\(.*|getenv.*|x{3,}|\*{3,}|\.{3,}|changeme|placeholder|example|"
    r"redacted|\[redacted.*|none|null|undefined|true|false|string|str|number|optional|required|z\.\w+\(?.*|string\(\)|"
    r"\w+\(\)|[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+|self\.\w+|this\.\w+|input|value|args?\b.*)$")


def shannon(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def looks_high_entropy(tok: str) -> bool:
    """A long mixed-alphabet token with high entropy (base64/base62 secret-like). Hex digests and UUIDs are excluded."""
    if len(tok) < 24 or _UUID.match(tok) or _HEX.match(tok):
        return False
    t = tok.strip("=")
    digits = sum(ch.isdigit() for ch in t)
    upper = sum(ch.isupper() for ch in t)
    lower = sum(ch.islower() for ch in t)
    if digits < 3 or upper < 2 or lower < 2:
        return False
    # path-like / word-like tokens (many separators) are not secrets
    if t.count("-") + t.count("_") + t.count("/") > len(t) / 6:
        return False
    return shannon(t) >= 4.2


_PROSE = re.compile(r"^[a-z]+$|^[A-Z][a-z]+$")


def _secretish(v: str) -> bool:
    """A value after "password is" that looks like a credential, not prose: not a plain lowercase / Capitalised word and not a
    file path or URL; contains a digit, a symbol, or mixed case beyond the first letter."""
    if _PROSE.match(v) or "/" in v or "\\" in v or v.startswith(("http", "[", "<", "$", "{")):
        return False
    return any(ch.isdigit() for ch in v) or any(not ch.isalnum() for ch in v) or any(ch.isupper() for ch in v[1:])


def _luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0


@dataclass
class ScrubResult:
    text: str
    hits: Counter = field(default_factory=Counter)

    @property
    def secret(self) -> bool:
        """Any SECRET-class hit: the text must not be used for sft / memory (drop the record)."""
        return any(k in SECRET_CATEGORIES for k, v in self.hits.items() if v)


SECRET_CATEGORIES = {"private_key", "jwt", "api_key", "bearer", "conn_string", "password", "card_number", "high_entropy",
                     "session_secret"}


def scrub(text: str | None, hits: Counter | None = None) -> ScrubResult:
    """Redact secrets / PII in ``text``. Counts per category are added to ``hits`` (if given) and returned."""
    local: Counter = Counter()
    if not text:
        return ScrubResult("" if text is None else text, local)
    s = text

    for cat, rx in _SECRET_PATTERNS:
        if cat == "password_phrase":
            def _pp(m: re.Match) -> str:
                v = m.group("v").rstrip(".")
                if _PLACEHOLDER.match(v) or not _secretish(v):
                    return m.group(0)
                local["password"] += 1
                return f"{m.group('k')}{m.group('q')}[REDACTED:password]"
            s = rx.sub(_pp, s)
        elif cat == "password":
            def _pw(m: re.Match) -> str:
                v = m.group("v")
                if _PLACEHOLDER.match(v) or v.lower().startswith(("ci-only-", "local-dev-")) and shannon(v) < 3.0:
                    return m.group(0)
                local["password"] += 1
                return f"{m.group('k')}{m.group('q')}[REDACTED:password]"
            s = rx.sub(_pw, s)
        elif cat == "card_number":
            def _card(m: re.Match) -> str:
                d = re.sub(r"\D", "", m.group(0))
                if 13 <= len(d) <= 19 and _luhn_ok(d) and len(set(d)) > 1:
                    local["card_number"] += 1
                    return "[REDACTED:card]"
                return m.group(0)
            s = rx.sub(_card, s)
        else:
            s, n = rx.subn(f"[REDACTED:{cat}]", s)
            if n:
                local[cat] += n

    def _ent(m: re.Match) -> str:
        if looks_high_entropy(m.group(0)):
            local["high_entropy"] += 1
            return "[REDACTED:high_entropy]"
        return m.group(0)
    s = _HIGH_ENTROPY_CAND.sub(_ent, s)

    for cat, rx in _PII_PATTERNS:
        s, n = rx.subn(f"[REDACTED:{cat}]", s)
        if n:
            local[cat] += n

    s, n = SENSITIVE_PATH_RE.subn(lambda m: m.group(0)[:1] + "[SENSITIVE_PATH]" if m.group(0)[:1] in "\\/ \"'`=(_-" else "[SENSITIVE_PATH]", s)
    if n:
        local["sensitive_path_mention"] += n

    if hits is not None:
        hits.update(local)
    return ScrubResult(s, local)


def scrub_text(text: str | None, hits: Counter | None = None) -> str:
    return scrub(text, hits).text
