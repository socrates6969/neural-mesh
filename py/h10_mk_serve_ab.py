"""h10 helper: derive src/bin/h10_serve_ab.rs from the (untouched) src/bin/g6_serve_ab.rs by text substitution + the real-payload block
in py/h10_serve_real_block.rs.  Run once; the generated .rs is the artifact."""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
s = open(os.path.join(ROOT, "src/bin/g6_serve_ab.rs"), encoding="utf8").read()
a = s.index('#[path = "g6_common/mod.rs"]')
HDR = """//! H10 SERVE A/B on REAL-TEXT request payloads: the frozen HEAD `neural-mesh serve` (baseline exe) vs the working-tree one.
//! Derived from g6_serve_ab (copy; g6 files are untouched).  Differences: the request script (real_script) is built from REAL text:
//! route/feedback features are the real keyword features of real doc sentences (py/h10_replay.py fixtures), some requests carry whole real
//! sentences (quotes, dashes, unicode) as feature strings, feedback rewards follow a SIMULATED competence that is tied to the REAL topic
//! clusters (best agent = cluster % 6).  Hostile/error/boundary lines and the mutation fuzz are kept (fuzz is applied to the real requests).
//!   equiv  - byte-for-byte reply streams, exit codes and final state files, 14 scenarios + base-vs-base controls + harness self-test.
//!   bench  - decisions/s with the real-text payloads (lockstep/pipelined/stdio), ABAB interleaved.
//! Usage: h10_serve_ab equiv [--base EXE] [--new EXE] [--quick] [--out FILE] | bench [--base EXE] [--new EXE] [--secs S] [--reps N] [--filter STR] [--out FILE]
//! Baseline exe: `bash results/h10_build_baseline.sh` (HEAD 52f936c via git archive) -> $CARGO_TARGET_DIR/h10/neural-mesh-base.exe

"""
s = HDR + s[a:]
s = s.replace("use serde_json::{Value, json};", "use serde_json::{Value, json};\nuse std::sync::OnceLock;", 1)
s = s.replace("script(0x6606,", "real_script(0x6606,").replace("script(0x6607,", "real_script(0x6607,")
old = 'let at = mutated.windows(3).position(|w| w == b"\\"kw").unwrap() + 2;'
assert old in s
s = s.replace(old, 'let at = mutated.windows(9).position(|w| w == b"mid-stats").unwrap() + 8;')
s = s.replace('join("g6")', 'join("h10")')
s = s.replace("build baseline with results/g6_build_baseline.sh", "build baseline with results/h10_build_baseline.sh")
s = s.replace("const TASKS: usize = 240;", "const TASKS: usize = 600;")
a = s.index("fn task_feats(t: usize)")
b = s.index("fn warm_line()")
NEW = '''fn best_agent(t: usize) -> usize {
    real().cluster[t % real().tasks.len()] % 6
}

fn feat_json(t: usize) -> String {
    let r = real();
    serde_json::to_string(&r.tasks[t % r.tasks.len()]).unwrap()
}

fn route_line(t: usize) -> String {
    format!("{{\\"cmd\\":\\"route\\",\\"features\\":{}}}\\n", feat_json(t))
}
fn feedback_line(t: usize, agent: usize) -> String {
    let r = if agent == best_agent(t) { 0.9 } else { 0.1 };
    format!("{{\\"cmd\\":\\"feedback\\",\\"features\\":{},\\"agent\\":\\"a{agent}\\",\\"reward\\":{r}}}\\n", feat_json(t))
}
'''
s = s[:a] + NEW + s[b:]
s = s.replace("feedback_line(tt, (task_feats(tt).0) % 6)", "feedback_line(tt, best_agent(tt))")
s = s.replace("task_feats(t).0 % 6", "best_agent(t)")
assert "task_feats" not in s
s += open(os.path.join(ROOT, "py/h10_serve_real_block.rs"), encoding="utf8").read()
open(os.path.join(ROOT, "src/bin/h10_serve_ab.rs"), "w", encoding="utf8").write(s)
print("ok", len(s))
