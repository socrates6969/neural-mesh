"""(iii) fire rate, added tokens and hook latency of the codemem UserPromptSubmit hook on REAL user prompts
(the same prompt loader as hooks/measure_recall.py: ingest state + raw transcripts, scrubbed).
  python bench/codemem/measure_fire.py [--n 400] [--latency 40]
Scenario 1: each prompt with its recorded cwd (what the hook sees live). Scenario 2: every prompt with cwd = each configured repo in turn is NOT
done (it would measure the gate on the wrong repo); instead scenario 2 pretends cwd = freight-recovery-build for prompts recorded in freight-recovery
sessions (the TS monorepo lives in that worktree) and cwd = neural-mesh for the rest. Tokens are chars/4 (labelled estimate) AND chars/3 (code upper bound).
The hook is run as the real exe, one process per prompt, with NEURAL_MESH_CODEMEM_LOG=off so nothing is appended to hook.jsonl."""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "hooks"))
import measure_recall as mr  # noqa: E402
EXE = ROOT / "target" / "release" / "codemem.exe"

def run(prompt, cwd):
    t = time.perf_counter()
    r = subprocess.run([str(EXE), "hook"], input=json.dumps({"prompt": prompt, "cwd": cwd, "session_id": "measure"}), capture_output=True, text=True, encoding="utf8", env=dict(os.environ, NEURAL_MESH_CODEMEM_LOG="off"), timeout=20)
    return r.stdout, (time.perf_counter() - t) * 1000, r.returncode

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); ap.add_argument("--latency", type=int, default=40); a = ap.parse_args()
    rows = mr.load_prompts(a.n)
    print("real prompts:", len(rows))
    out = {}
    for name, remap in (("recorded_cwd", False), ("repo_cwd", True)):
        fired, chars, ms, bad, gate_pass = [], [], [], 0, 0
        for r_ in rows:
            cwd, prompt = r_["cwd"], r_["prompt"]
            c = cwd
            if remap:
                c = "C:/Users/mariu/ventures/freight-recovery-build" if "freight" in cwd.lower() else "C:/Users/mariu/neural-mesh"
            g = subprocess.run([str(EXE), "gate", prompt[:700].replace(chr(10), " ")], capture_output=True, text=True)
            try: gate_pass += json.loads(g.stdout)["fire"]
            except ValueError: pass
            o, t, rc = run(prompt, c); ms.append(t); bad += rc != 0
            if o.strip(): fired.append(prompt[:90].replace("\n", " ")); chars.append(len(o))
        out[name] = {"n": len(rows), "gate_pass": gate_pass, "gate_pass_rate": round(gate_pass / len(rows), 4), "fires": len(fired), "fire_rate": round(len(fired) / len(rows), 4), "exit_nonzero": bad,
                     "chars_when_fired_mean": round(statistics.mean(chars), 1) if chars else 0, "tokens_est_chars4_mean": round(statistics.mean(chars) / 4, 1) if chars else 0,
                     "tokens_est_chars3_max": round(max(chars) / 3, 1) if chars else 0, "added_tokens_per_prompt_chars4": round(sum(chars) / 4 / len(rows), 2),
                     "e2e_ms_p50": round(pct(ms, .5)), "e2e_ms_p95": round(pct(ms, .95)), "e2e_ms_max": round(max(ms)), "examples": fired[:6]}
    print(json.dumps(out, indent=1))
main()
