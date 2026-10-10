"""Outcome labeler v2 (R122): labels each dispatch / continuation from EXPLICIT structural signals instead of only report text.

Why: the v1 labeler (extract.classify) read text cues in the agent's FINAL report. For a chain (a dispatch plus its SendMessage continuations) the
final report is the same for every segment, and most outcomes of a segment are only visible from what the leader did NEXT:
* the next message to the same agent says "you hit your turn limit" / "terminated by an API rate limit"  -> this segment was cut off (re-dispatch, env);
* a builder "fix round N" message (or a "Fix ... defects D1..D6" dispatch) follows a builder / tester / reviewer segment and names defects of that
  role's kind (tester defects D-n, reviewer findings F-n, CI)  -> this segment found defects ("defects");
* the segment's own message starts as a fix task ("Builder fix round 3", "CI diagnosis", "Small follow-up ... CI result")  -> fix-round.
Text cues are applied to the LAST segment of a chain only (that is the only segment whose report is the one stored), and agent-interruption cues are
phrased as interruptions (a product report that mentions "rate limiting" is not an environment failure).

Precedence: re-dispatch (env) > blocked > fix-round > defects > success.  Everything here is a HEURISTIC proxy; agreement with the 53 hand labels
(tests/fixtures/hand_labels.json, written by the evaluator from the run dirs' status.md / mailbox.md) is measured by `python -m ingest.labeler validate`.
The rules were written after reading those 53 labels, so that agreement is IN-SAMPLE; the number to trust is the one on the next hand-labelled batch.

CLI (no ingest/__main__ edit needed):  python -m ingest.labeler validate | relabel [--write]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .extract import parse_ts

HAND_LABELS = Path(__file__).parent / "tests" / "fixtures" / "hand_labels.json"
HAND_TO_V = {"passed": "success", "defects": "defect", "fix-round": "fix-round", "re-dispatch": "re-dispatch", "blocked": "blocked"}

# --- the agent was interrupted (environment), NOT a product topic: phrased as an interruption
_ENV_REPORT = re.compile(r"(?i)\b(?:maximum number of turns|max[_ ]turns|turn limit|usage limit|weekly limit|hit your limit|you'?ve hit|overloaded)\b|(?-i:\bAPI Error[: ])"
                         r"|\bterminated by\b|\bHTTP 429\b|\b429\b|rate.?limit(?:ed)?\s+(?:reached|hit|exceeded|error)|\bBUILDER_BLOCK\b"
                         r"|\bran out of (?:turns|context)\b|\bcontext (?:limit|window) (?:reached|exceeded)\b"
                         r"|worktree\b.{0,60}\b(?:block|isolat|lost|broken|fail)|\b(?:isolat|harness)\w*\b.{0,40}\bworktree")
# --- a message that RESUMES an interrupted agent: the PREVIOUS segment was cut off
_ENV_RESUME = re.compile(r"(?i)\byou hit (?:your|the)\b.{0,25}\bturn limit|\bturn limit\b|\bterminated by an? (?:API )?(?:rate limit|error)|\bHTTP 429\b|\b429\b"
                         r"|\busage[- ]limit|\bcut ?off\b|\bresum(?:e|ing) after\b|\bcontext (?:limit|window)\b")
_BLOCKED = re.compile(r"(?i)\b(?:SHIP|BUILD|TEST)_BLOCKED\b|\bpermission (?:denied|classifier|prompt)\b|\b(?:push|command|tool call)s? (?:was |were )?denied\b|\bdenied by\b"
                      r"|\bblocked (?:by|on)\b|\bcannot proceed\b|\bawaiting (?:user|approval|your)\b|\bneed(?:s)? your (?:approval|go-ahead|decision)\b"
                      r"|\bhook blocked\b|\bwas blocked\b|\bstopp(?:ed|ing) (?:and|to) report\b")
# --- explicit verdict tokens of a defect-finding report (applied to the last segment only, never to scriber/shipper/planner/distiller)
_DEFECT_VERDICT = re.compile(r"\bTEST_BLOCK\b|\bDEFECTS?_FOUND\b|\bCHANGES[_ ]REQUESTED\b|\bVerdict:?\s*\W*(?:REJECT(?:ED)?|BLOCK(?:ED)?|FAIL)\b")
# explicit hand-back verdicts kept as signals (a gate-negative research result such as "NOT promoted" is an honest finding, not a defect of the work)
_POSITIVE = re.compile(r"\b(?:SPEC_READY|BUILD_COMPLETE|TESTS?_AUTHORED|TEST_COMPLETE|TEST_PASS|DOCS_COMPLETE|SHIPPED|SHIP_COMPLETE|CI_GREEN|REVIEW_PASS)\b|\bVerdict:?\s*\W*PASS\b")
_GATE_NEG = re.compile(r"\bNOT promoted\b|\bREJECTED\b")
_DEFECT_COUNT = re.compile(r"(?<!had )(?<!was )(?<!were )\b[1-9]\d* (?:tests? )?(?:failed|failing)\b(?! (?:in|on) round)")
# --- the leader's message starts as a fix task
_FIX_START = re.compile(r"(?is)^\W*(?:(?:builder|tester|scriber)\s+)?(?:fix(?:\s+round)?\b|small follow-up\b|ci\s+(?:diagnosis|round|result|fail|follow)|round\s+\d+\s+fix)")
_FIX_DESC = re.compile(r"(?i)^\s*(?:fix|hotfix)\b|\bfix(?:es)? (?:round|defects?|ci|review)\b")
_NOTE = re.compile(r"(?is)^\W*(?:note from|no reply needed|please stop|fyi\b|queued)|\bno reply needed\b")
_ROLE_RX = re.compile(r"(?i)(?:acting as|you are)(?: the)? (?:statsclaw )?\W*(builder|tester|reviewer|planner|scriber|shipper|distiller)\b")
_RUNKEY = re.compile(r"REQ-\d{8}-[\w-]+|feat/[\w-]+")
ROLES = {"builder", "tester", "reviewer", "planner", "scriber", "shipper", "distiller"}


def role_of(agent: str, head_prompt: str) -> str:
    a = (agent or "").lower()
    a = a.split(":")[-1].removeprefix("fr-")
    if a in ("security-reviewer", "code-reviewer"):
        a = "reviewer"
    if a in ROLES:
        return a
    m = _ROLE_RX.search((head_prompt or "")[:500])
    return m.group(1).lower() if m else "general"


def fix_origin(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"independent reviewer|from the (?:independent )?reviewer|review findings|review\.md", t):
        return "review"
    if re.search(r"\(ci only\)|ci fail|web-ci|github actions|ci round", t):
        return "ci"
    return "test"


def _run_keys(items) -> set[str]:
    keys = set()
    for it in items:
        keys.update(_RUNKEY.findall((it.prompt or "")[:1500] + " " + (it.description or "")))
    return keys


def _same_run(a: set[str], b: set[str]) -> bool:
    return not a or not b or bool(a & b)


def label_items(items: list) -> dict[str, tuple[str, list[str]]]:
    """items: DispatchItem list (all sessions). Returns {item id: (outcome, signals)}."""
    out: dict[str, tuple[str, list[str]]] = {}
    by_session = defaultdict(list)
    for it in items:
        by_session[it.session].append(it)
    for sess_items in by_session.values():
        chains: dict[str, list] = defaultdict(list)
        for it in sorted(sess_items, key=lambda x: (x.ts or "", x.id)):
            chains[it.chain or it.id].append(it)
        meta, chain_ids = {}, {}
        for cid, ch in chains.items():
            chain_ids[cid] = {x.id for x in ch}
            head = ch[0]
            meta[cid] = {"role": role_of(head.agent, head.prompt), "runs": _run_keys(ch)}
        # all builder fix items of the session, with origin
        fixes = []
        for cid, ch in chains.items():
            for it in ch:
                if meta[cid]["role"] in ("builder", "general") and (_FIX_START.search(it.prompt or "") or _FIX_DESC.search(it.description or "") and it.kind == "dispatch"):
                    fixes.append((parse_ts(it.ts), fix_origin((it.prompt or "")[:500] + " " + (it.description or "")), meta[cid]["runs"], it.id))
        by_role_ts = defaultdict(list)       # role -> [(ts, runs, id)]
        for cid, ch in chains.items():
            for it in ch:
                by_role_ts[meta[cid]["role"]].append((parse_ts(it.ts), meta[cid]["runs"], it.id))
        for cid, ch in chains.items():
            role, runs = meta[cid]["role"], meta[cid]["runs"]
            real = [k for k, x in enumerate(ch) if not (x.kind == "continuation" and _NOTE.search((x.prompt or "")[:200]))]
            last_real = real[-1] if real else len(ch) - 1       # queued notes carry no report of their own: the last REAL segment owns the stored report
            for i, it in enumerate(ch):
                last = i == last_real
                nxt = next((x for x in ch[i + 1:] if not _NOTE.search((x.prompt or "")[:200])), None) if i < last_real else None
                t0 = parse_ts(it.ts)
                msg = it.prompt or ""
                tail = (it.report or "")[-700:] + "\n" + (it.result_head or "")
                sig: list[str] = []
                if last:
                    if it.later.get("no_report"):
                        sig.append("no_final_report")
                    if it.signals.get("turn_limit"):
                        sig.append("turn_limit")
                    if it.signals.get("api_error"):
                        sig.append("api_error")
                    if it.later.get("redispatched"):
                        sig.append("redispatched_later")
                    if it.later.get("result_error"):
                        sig.append("result_error")
                    if _ENV_REPORT.search(tail):
                        sig.append("redispatch_cue")
                if nxt is not None and _ENV_RESUME.search((nxt.prompt or "")[:300]):
                    sig.append("resumed_after_interrupt")
                is_fix = bool(_FIX_START.search(msg[:120]) or (it.kind == "dispatch" and _FIX_DESC.search(it.description or ""))) and role in ("builder", "tester", "general")
                if is_fix:
                    sig.append("fix_task")
                if last and _BLOCKED.search(tail):
                    sig.append("blocked_cue")
                if it.kind == "continuation" and _NOTE.search(msg[:200]):
                    out[it.id] = ("success", ["note"])        # a queued note is not a unit of work with its own outcome
                    continue
                rep0 = it.report or ""
                if last and _POSITIVE.search(rep0[:400] + " " + rep0[-600:]):
                    sig.append("verdict_positive")
                if last and _GATE_NEG.search(rep0[:400] + " " + rep0[-600:]):
                    sig.append("verdict_gate_negative")
                s = set(sig)
                if s & {"no_final_report", "turn_limit", "api_error", "redispatched_later", "result_error", "redispatch_cue", "resumed_after_interrupt"}:
                    out[it.id] = ("re-dispatch", sig)
                    continue
                if "blocked_cue" in s:
                    out[it.id] = ("blocked", sig)
                    continue
                if is_fix:
                    out[it.id] = ("fix-round", sig)
                    continue
                # defects: a builder fix message of this role's origin follows (structural), or an explicit verdict token in the stored report
                defect = False
                if role == "builder":
                    # a defect/review fix to the same agent (next message of the chain) or a later builder fix dispatch of the same run
                    defect = any(ft > t0 and o in ("test", "review") and fid != it.id and (fid in chain_ids[cid] or (fr and runs and fr & runs))
                                 for ft, o, fr, fid in fixes)
                elif role in ("tester", "reviewer"):
                    want = "test" if role == "tester" else "review"
                    nxt_same = min((x[0] for x in by_role_ts[role] if x[0] > t0 and _same_run(runs, x[1])), default=float("inf"))
                    defect = any(t0 < ft < nxt_same and o == want and _same_run(runs, fr) for ft, o, fr, _ in fixes)
                if defect:
                    sig.append("defects_followed")
                rep = it.report or ""
                if last and role not in ("scriber", "shipper", "planner", "distiller") and (_DEFECT_VERDICT.search(rep[:300] + " " + rep[-500:]) or (_DEFECT_COUNT.search(rep[-300:]) and "verdict_positive" not in sig)):
                    sig.append("defect_token")
                    defect = True
                out[it.id] = ("defect" if defect else "success", sig)
    return out


# ---------------------------------------------------------------- validation + relabel
def load_hand(path: Path = HAND_LABELS) -> dict[str, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: HAND_TO_V[v["outcome"]] for k, v in raw.items() if not k.startswith("_") and v.get("outcome") in HAND_TO_V}


def confusion(pairs: list[tuple[str, str]]) -> dict:
    """pairs = (hand, predicted). Returns counts, agreement and the matrix."""
    classes = ["success", "defect", "fix-round", "re-dispatch", "blocked"]
    mat = {h: {p: 0 for p in classes} for h in classes}
    for h, p in pairs:
        mat[h][p] += 1
    n = len(pairs)
    agree = sum(h == p for h, p in pairs)
    bin_agree = sum((h == "success") == (p == "success") for h, p in pairs)
    return {"n": n, "exact": agree, "success_vs_not": bin_agree, "matrix_rows_hand_cols_pred": mat}


def build_items(cfg, now: float | None = None):
    import time
    from . import extract as ex
    from .reader import load_all_states
    sessions = ex.group_sessions(load_all_states(cfg.files_dir))
    now = now or time.time()
    return [d for s in sessions for d in ex.dispatch_items(s, now, cfg.settle_minutes * 60)]


def validate(cfg, hand_path: Path = HAND_LABELS) -> dict:
    hand = load_hand(hand_path)
    items = build_items(cfg)
    new = label_items(items)
    by6 = {it.id[-6:]: it for it in items}
    old = {}
    from .util import iter_jsonl
    for r in iter_jsonl(cfg.outcomes):
        old[r["id"][-6:]] = r["outcome"]
    keys = [k for k in hand if k in by6]
    v1 = [(hand[k], old.get(k, "success")) for k in keys]
    v2 = [(hand[k], new[by6[k].id][0]) for k in keys]
    res = {"labelled": len(keys), "v1": confusion(v1), "v2": confusion(v2), "v2_errors": []}
    for k in keys:
        if hand[k] != new[by6[k].id][0]:
            res["v2_errors"].append({"id": k, "hand": hand[k], "v2": new[by6[k].id][0], "signals": new[by6[k].id][1], "role": role_of(by6[k].agent, by6[k].prompt)})
    return res


def relabel(cfg, write: bool = False) -> dict:
    """Recompute every outcome's label (+reward, signals) from the saved per-file states. Rows keep id/ts/judge; the old file is backed up."""
    from . import extract as ex
    from .pipeline import _reward
    from .util import atomic_write_text, iter_jsonl
    new = label_items(build_items(cfg))
    rows = list(iter_jsonl(cfg.outcomes))
    changed = Counter()
    out_rows = []
    for r in rows:
        if r["id"] in new:
            oc, sig = new[r["id"]]
            keep = [s for s in (r.get("signals") or []) if s == "prompt_secret_dropped"]
            if oc != r["outcome"]:
                changed[f'{r["outcome"]}->{oc}'] += 1
            r = dict(r, outcome=oc, signals=sig + keep, reward=_reward(oc, r.get("judge_score")), label_version=2)
        out_rows.append(r)
    res = {"rows": len(rows), "relabelled": sum(1 for r in rows if r["id"] in new), "changed": sum(changed.values()), "transitions": dict(changed),
           "counts_before": dict(Counter(r["outcome"] for r in rows)), "counts_after": dict(Counter(r["outcome"] for r in out_rows)), "written": False}
    if write:
        import shutil
        bak = cfg.outcomes.with_name("outcomes.v1.jsonl")
        if not bak.exists():
            shutil.copy2(cfg.outcomes, bak)
        atomic_write_text(cfg.outcomes, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows))
        res["written"] = True
        res["backup"] = str(bak)
    return res


def main(argv=None) -> int:
    from .config import Config
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "validate"
    cfg = Config()
    if cmd == "validate":
        print(json.dumps(validate(cfg), indent=1))
        return 0
    if cmd == "relabel":
        print(json.dumps(relabel(cfg, write="--write" in argv), indent=1))
        return 0
    print("usage: python -m ingest.labeler validate | relabel [--write]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
