"""bench/common.py - small shared helpers: line-ending-independent file hashes, canonical JSON hashes, git provenance, PREREG."""
import hashlib, json, os, pathlib, subprocess

HERE = pathlib.Path(__file__).resolve().parent
NM = HERE.parent
PREREG_PATH = HERE / "PREREG.json"
SPLITS = ("tune", "held", "conf")
AXES = ("math", "physics", "code", "research_public", "research_private")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha(path):
    """sha256 of a text file with CRLF normalised to LF, so a core.autocrlf checkout does not change the key"""
    return sha_bytes(pathlib.Path(path).read_bytes().replace(b"\r\n", b"\n"))


def json_sha(obj):
    return sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode())


def axis_of(item):
    return f"research_{item['sub']}" if item["cat"] == "research" else item["cat"]


def battery_path(split):
    return HERE / f"battery_{split}.json"


def load_battery(split):
    j = json.load(open(battery_path(split), encoding="utf8"))
    return j["header"], j["items"]


def load_prereg(path=None):
    p = pathlib.Path(path or PREREG_PATH)
    return (json.load(open(p, encoding="utf8")), file_sha(p)) if p.exists() else (None, None)


def git_info():
    def g(*a):
        try: return subprocess.run(["git", *a], cwd=str(NM), capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception: return ""
    return dict(git_sha=g("rev-parse", "HEAD"), git_dirty=bool(g("status", "--porcelain", "--untracked-files=no")))
