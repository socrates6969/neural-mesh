"""Registry lookup evaluation: 50 'I need to ...' queries (authored before any scoring) -> acceptable capability names.
Arms: dense (MiniLM), bm25, hybrid z-fusion .7, adaptive alpha (R2), + variants. Metric hit@1, hit@3, MRR (rank of first acceptable entry).
No tuning is done on these queries (all arms use frozen parameters); statistics are paired exact sign tests over queries, Holm across arms."""
import json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "b1"))
from lib import BM25, toks, z
from embed import embed
from scipy import stats

REG = "C:/Users/mariu/experiments/mesh-memory/registry/capabilities.json"
Q = [
("run a backtest of a trading strategy on historical prices", {"trader-backtest", "backtest-engineer"}),
("explain why a trade signal was generated", {"trader-explain"}),
("measure portfolio risk and value at risk", {"trader-risk", "risk-analyst"}),
("search my stored memory for earlier decisions", {"memory-search", "recall", "memory-specialist"}),
("start a multi-agent swarm", {"swarm-init", "swarm", "coordinator"}),
("watch a live event stream", {"monitor-stream", "watch"}),
("diagnose whether my installation is healthy", {"ruflo-doctor", "ruflo-status"}),
("automate a browser login flow", {"browser-login", "browser-auth-flow"}),
("scrape data from a web page", {"browser-scrape", "browser-extract"}),
("record and replay a browser session", {"browser-record", "browser-replay"}),
("compare screenshots to find visual differences", {"browser-screenshot-diff"}),
("create an architecture decision record", {"adr-create"}),
("review an architecture decision", {"adr-review"}),
("detect personal information in text", {"pii-detect"}),
("scan a prompt for injection attacks", {"safety-scan", "safety-specialist"}),
("track token cost per agent", {"cost-track", "cost-report", "cost-analyst"}),
("write a database migration", {"migrate-create", "migration-engineer"}),
("validate a plugin manifest", {"validate-plugin", "plugin-validator"}),
("generate unit tests with test driven development", {"tdd-workflow", "test-gaps", "tester", "testgen"}),
("version control tool", {"git"}),
("build and test rust code", {"cargo", "rustc"}),
("manage pull requests from the terminal", {"gh", "github"}),
("run applications in containers", {"docker"}),
("install javascript packages", {"npm"}),
("very fast python package manager", {"uv"}),
("convert or resize images from the command line", {"convert"}),
("secure remote shell login", {"ssh"}),
("run a linux environment on windows", {"wsl"}),
("python library for numerical arrays", {"numpy"}),
("python machine learning algorithms library", {"scikit-learn"}),
("scientific computing and linear algebra in python", {"scipy"}),
("tokenizer library for language models", {"tokenizers"}),
("run neural network models for inference", {"onnxruntime"}),
("look up up-to-date library documentation for developers", {"context7"}),
("end to end browser automation testing", {"playwright"}),
("issue tracking and project management integration", {"linear", "asana", "gitlab", "github"}),
("provision infrastructure as code", {"terraform"}),
("build an MCP server", {"build-mcp-server", "mcp-server-dev"}),
("create a new skill", {"skill-creator", "skill-development"}),
("turn a short product idea into a detailed specification", {"planner"}),
("review a pull request for code quality", {"code-review", "pr-review-toolkit", "code-reviewer"}),
("extract entities and relations into a knowledge graph", {"kg-extract", "graph-navigator", "kg-traverse"}),
("semantic vector search over embeddings", {"vector-search", "vector-embed", "embeddings"}),
("schedule a recurring background job", {"cron-schedule", "loop-worker", "ruflo-schedule"}),
("connect agents running on different machines", {"federation-init", "federation-coordinator", "cross-host-federation"}),
("run an isolated sandbox agent in WebAssembly", {"wasm-agent", "wasm-specialist"}),
("generate music from lyrics", {"music-generate", "music"}),
("match candlestick patterns in market data", {"market-pattern", "market-ingest", "data-engineer"}),
("send chat messages through telegram", {"telegram"}),
("hook automation lifecycle events", {"Hooks Automation", "hook-development", "hookify"}),
("coordinate several coding agents with consensus", {"consensus-coordinator", "agent-consensus-coordinator", "queen-coordinator"}),
]

Q2 = [  # authored AFTER the registry enrichment, different wording, never used for any decision (clean held-out v2)
("format my rust source files", {"rustfmt", "cargo-fmt"}), ("check rust code for common mistakes", {"cargo-clippy", "clippy-driver"}),
("download a file over http", {"curl"}), ("edit and browse source code in an editor", {"code"}), ("compress files into an archive", {"tar"}),
("call a web api endpoint from a script", {"curl"}), ("automate tasks with a windows shell script", {"powershell"}), ("step through a rust program in a debugger", {"rust-gdb", "rust-lldb", "rust-gdbgui"}),
("code completion and diagnostics for rust in an editor", {"rust-analyzer"}), ("interactive python interpreter", {"python"}), ("serialize messages with protocol buffers", {"protobuf"}),
("statistical distributions and tests", {"scipy"}), ("openai coding agent in the terminal", {"codex"}), ("ask claude from the command line", {"claude"}),
("update and switch rust toolchain versions", {"rustup"}), ("generate api documentation for a rust crate", {"rustdoc"}), ("send discord messages", {"discord"}),
("firebase backend services", {"firebase"}), ("php laravel framework helper", {"laravel-boost"}), ("send and read imessages", {"imessage"}), ("gitlab merge requests", {"gitlab"}),
("semantic code navigation with a language server", {"serena"}), ("guided workflow for developing a new feature", {"feature-dev"}), ("keep iterating a prompt in a loop until done", {"ralph-loop"}),
("explanatory output style for learning", {"explanatory-output-style", "learning-output-style"}), ("lint check for undefined behavior in rust", {"cargo-miri"}),
]

def load():
    d = json.load(open(REG)); ents = d["entries"]
    texts = [f"{e['kind']} {e['name']}: {e['desc']}" for e in ents]
    cache = "C:/Users/mariu/experiments/mesh-memory/registry/emb.npy"
    E = np.load(cache) if os.path.exists(cache) and len(np.load(cache)) == len(texts) else None
    if E is None:
        E = embed(texts); np.save(cache, E)
    return ents, texts, E

def ranks(ents, scores, accept):
    order = np.argsort(-scores)
    acc = {a.lower() for a in accept}
    for r, i in enumerate(order[:200]):
        if ents[i]["name"].lower() in acc: return r + 1
    return 999

def adaptive(bm, qtext, dense, lo=.3, hi=1.0, pw=2.0):
    b = bm.score(qtext); seen = {t for t in toks(qtext) if t in bm.post}; den = sum(bm.idf[t] for t in seen) * (bm.k1 + 1)
    f = (b.max() / den) if den > 0 else 0.0; a = hi - (hi - lo) * min(1.0, f) ** pw
    return a * z(dense) + (1 - a) * z(b)

def run(arms, ents, texts, E, queries=Q):
    bm = BM25(texts); QE = embed([q for q, _ in queries]); res = {}
    for name, fn in arms.items():
        res[name] = np.array([ranks(ents, fn(bm, q, E @ QE[i], i), acc) for i, (q, acc) in enumerate(queries)])
    return res

def summarize(res):
    for n, r in res.items():
        print(f"  {n:22s} hit@1 {np.mean(r==1):.3f}  hit@3 {np.mean(r<=3):.3f}  hit@10 {np.mean(r<=10):.3f}  MRR {np.mean(1.0/r):.3f}")

def sign_p(a, b):
    """exact two-sided sign test on paired reciprocal ranks"""
    d = (1.0 / a) - (1.0 / b); w, l = int((d > 1e-12).sum()), int((d < -1e-12).sum())
    return stats.binomtest(w, w + l, 0.5).pvalue if w + l else 1.0, w, l

if __name__ == "__main__":
    ents, texts, E = load()
    arms = {"dense": lambda bm, q, d, i: d, "bm25": lambda bm, q, d, i: bm.score(q), "hybrid.7": lambda bm, q, d, i: .7 * z(d) + .3 * z(bm.score(q)), "adaptive": lambda bm, q, d, i: adaptive(bm, q, d)}
    res = run(arms, ents, texts, E); summarize(res)
    print("-- clean held-out Q2 (n=%d)" % len(Q2)); r2 = run(arms, ents, texts, E, Q2); summarize(r2)
    for a in ("dense", "bm25", "adaptive"):
        p, w, l = sign_p(res[a], res["hybrid.7"]); print(f"  {a} vs hybrid.7: W/L {w}/{l} sign-p {p:.3g}")
    miss = [(Q[i][0], sorted(Q[i][1])) for i in range(len(Q)) if res["adaptive"][i] > 3]
    print("adaptive misses (rank>3):", len(miss)); [print("   ", m) for m in miss[:12]]
