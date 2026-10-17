"""E3 (hive exp-router): router feature hygiene, prior variants and a new learner, scored prequentially on a pre-registered time split.

Pre-registration: C:/Users/mariu/ruv/hive/findings/exp-router.md. Split: kind=dispatch rows sorted by (ts, id); dispatch index < 60% = tune,
the rest = held-out. Every system runs online over the whole stream (predict, then feed) and is scored on the SAME rows (not an env failure,
>= 2 agents seen so far globally). Held-out numbers are produced only by `--split held`, run once on the frozen candidate.

CLI: python -m ingest.router_e3 --outcomes FILE --exe neural-mesh.exe --split tune|held [--systems a,b] [--json OUT]
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .extract import STOP, features as features_v1
from .meshio import clean_agent, is_env_failure
from .shadow import FR_ROSTER, RULE, mcnemar_exact_p, rule_pick, wilson
from .util import iter_jsonl

TUNE_FRAC = 0.6

# ---- feature hygiene (H1/H2) -------------------------------------------------------------------------------------------------------------

_URL = re.compile(r"https?://\S+")
_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|~[\\/]|(?<![\w.])\.{0,2}/(?=\w))[^\s'\"`,;()<>|]*")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
USERNAMES = {"mariu", "users", "appdata", "socrates6969"}
_PHASE = {kw: agent.replace("fr-", "") for agent, kws in RULE for kw in kws}


def clean_text(text: str) -> str:
    """Remove URLs, file-system paths and UUIDs before tokenising (they carry the user name and machine layout, not the task)."""
    t = _URL.sub(" ", str(text or ""))
    t = _UUID.sub(" ", t)
    return _PATH.sub(" ", t)


def features_clean(text: str, k: int = 24) -> list[str]:
    """extract.features minus noise: no path segments, user names, hashes or any token containing a digit (round ids, commit hashes, sizes)."""
    toks = [t for t in re.split(r"[^a-z0-9]+", clean_text(text).lower())
            if len(t) >= 2 and t not in STOP and t not in USERNAMES and not any(c.isdigit() for c in t)]
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= k:
            break
    return out


def phase_verb(text: str) -> str:
    """H2: the first phase keyword (shadow.RULE vocabulary) in the cleaned text, as one feature; verb:none if there is none."""
    for tok in re.split(r"[^a-z0-9]+", clean_text(text).lower()):
        if tok in _PHASE:
            return "verb:" + _PHASE[tok]
    return "verb:none"


FEATURES = {
    "v1": lambda r: list(r.get("features") or []),                 # what the live ingest stored (= extract.features at ingest time)
    "v1text": lambda r: features_v1(r.get("task_text") or ""),     # same tokenizer recomputed from the stored task_text (control for H1)
    "clean": lambda r: features_clean(r.get("task_text") or ""),
    "clean_verb": lambda r: features_clean(r.get("task_text") or "", 23) + [phase_verb(r.get("task_text") or "")],
}


def feat_arg(fs: list[str]) -> str:
    return ",".join([f for f in fs if f and "," not in f][:24] or ["_none"])


# ---- data ----------------------------------------------------------------------------------------------------------------------------------

def load(path: Path) -> list[dict]:
    return load_rows(iter_jsonl(path))


def load_rows(rows) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r.get("ts") or "", r.get("id") or ""))
    d = [r for r in rows if r.get("kind") == "dispatch"]
    cut = int(len(d) * TUNE_FRAC)
    for i, r in enumerate(d):
        r["_i"] = i
        r["_split"] = "tune" if i < cut else "held"
        r["_agent"] = clean_agent(r)
        r["_proj"] = r.get("project") or "default"
    return d


def scored_mask(rows: list[dict]) -> list[bool]:
    seen: set[str] = set()
    out = []
    for r in rows:
        out.append(len(seen) >= 2 and not is_env_failure(r))
        seen.add(r["_agent"])
    return out


# ---- systems: each returns one prediction per row (None on rows that are not scored) -------------------------------------------------------

def _run(cmd: list[str]) -> dict | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def sys_majority(rows, mask, scope="global"):
    g, per = Counter(), defaultdict(Counter)
    out = []
    for r, m in zip(rows, mask):
        c = g if scope == "global" else (per[r["_proj"]] or g)
        out.append(c.most_common(1)[0][0] if m and c else None)
        g[r["_agent"]] += 1
        per[r["_proj"]][r["_agent"]] += 1
    return out


def sys_rule(rows, mask):
    return [rule_pick(r.get("task_text") or "") if m else None for r, m in zip(rows, mask)]


def sys_mesh(rows, mask, exe: Path, feats="v1", scope="global", late=0.0, prior_k=None, roster=False, seed0=7, extra=()):
    """The Rust Hebbian router via its CLI (same calls as ingest/router_eval.py), label feed, dispatch units."""
    fx = FEATURES[feats]
    work = Path(tempfile.mkdtemp(prefix="e3-"))
    states: dict[str, Path] = {}
    seen_all: list[str] = []
    seen_proj: dict[str, list[str]] = defaultdict(list)
    g = Counter()
    out = []
    knobs = ["--late-agent-init", str(late), *extra]
    try:
        for i, (r, m) in enumerate(zip(rows, mask)):
            agent, proj = r["_agent"], r["_proj"]
            fa = feat_arg(fx(r))
            state = work / "g.json" if scope == "global" else states.setdefault(proj, work / f"p{len(states)}.json")
            cands = list(seen_all) if scope == "global" else list(seen_proj[proj])
            if roster and scope != "global" and proj == "freight-recovery":
                cands += [a for a in FR_ROSTER if a not in cands]
            pick = None
            if m:
                if len(cands) >= 2:
                    cmd = [str(exe), "route", "--state", str(state), "--features", fa, "--candidates", ",".join(cands), "--seed", str(seed0 + i), *knobs]
                    if prior_k is not None:
                        cmd += ["--prior", rule_pick(r.get("task_text") or ""), "--prior-k", str(prior_k)]
                    pick = (_run(cmd) or {}).get("agent")
                elif len(cands) == 1:
                    pick = cands[0]
                if pick is None:
                    pick = g.most_common(1)[0][0]
            out.append(pick)
            fb = (["--agent", agent, "--reward", "0", "--failure-class", "env"] if is_env_failure(r) else ["--label", agent])
            _run([str(exe), "feedback", "--state", str(state), "--features", fa, *fb, *knobs])
            if agent not in seen_all:
                seen_all.append(agent)
            if agent not in seen_proj[proj]:
                seen_proj[proj].append(agent)
            g[agent] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def sys_nb(rows, mask, feats="clean", alpha=0.5, proj_w=3.0, margin=0.0, scope_cands="global", roster=False):
    """H4: online multinomial naive Bayes over cleaned tokens + the project as a weighted pseudo-token, class prior = counts so far.
    It leaves the majority class only when log P(best) - log P(majority) > margin (abstain-to-majority), so a rare agent must be supported by
    evidence, not by a single shared word. Candidates = agents seen so far (globally, or in the project plus its declared roster)."""
    fx = FEATURES[feats]
    cls = Counter()
    tok = defaultdict(Counter)        # agent -> token counts
    tot = Counter()
    vocab: set[str] = set()
    seen_proj: dict[str, list[str]] = defaultdict(list)
    out = []
    for r, m in zip(rows, mask):
        toks = fx(r)
        ptok = "proj:" + r["_proj"]
        pick = None
        if m and cls:
            if scope_cands == "global":
                cands = list(cls)
            else:
                cands = list(seen_proj[r["_proj"]])
                if roster and r["_proj"] == "freight-recovery":
                    cands += [a for a in FR_ROSTER if a not in cands]
                cands = cands or list(cls)
            n = sum(cls.values())
            V = len(vocab) + 1

            def score(a):
                s = math.log((cls[a] + alpha) / (n + alpha * len(cands)))
                den = tot[a] + alpha * V
                for t in toks:
                    s += math.log((tok[a][t] + alpha) / den)
                s += proj_w * math.log((tok[a][ptok] + alpha) / den)
                return s
            sc = {a: score(a) for a in cands}
            best = max(sc, key=sc.get)
            maj = max(cands, key=lambda a: cls[a])
            pick = best if sc[best] - sc[maj] > margin else maj
        out.append(pick if m else None)
        if not is_env_failure(r):
            a = r["_agent"]
            cls[a] += 1
            for t in toks:
                tok[a][t] += 1
                tot[a] += 1
                vocab.add(t)
            tok[a][ptok] += 1
            tot[a] += 1
            vocab.add(ptok)
        if r["_agent"] not in seen_proj[r["_proj"]]:
            seen_proj[r["_proj"]].append(r["_agent"])
    return out


# ---- scoring ------------------------------------------------------------------------------------------------------------------------------

def _sel(rows, mask, split, only_fr=False):
    return [i for i, (r, m) in enumerate(zip(rows, mask)) if m and r["_split"] == split and (r["_agent"] in FR_ROSTER or not only_fr)]


def metrics(rows, idx, pred) -> dict:
    y = [rows[i]["_agent"] for i in idx]
    c = [pred[i] == rows[i]["_agent"] for i in idx]
    n = len(idx)
    k = sum(c)
    per = defaultdict(lambda: [0, 0])
    for yy, cc in zip(y, c):
        per[yy][0] += cc
        per[yy][1] += 1
    ba = sum(a / b for a, b in per.values()) / len(per) if per else None
    lo, hi = wilson(k, n) if n else (0, 0)
    return {"n": n, "acc": round(k / n, 4) if n else None, "acc_ci95": [round(lo, 4), round(hi, 4)], "ba": round(ba, 4) if ba is not None else None,
            "recall": {a: f"{v[0]}/{v[1]}" for a, v in sorted(per.items(), key=lambda kv: -kv[1][1])}}


def paired(rows, idx, pa, pb, draws=20000, seed=0) -> dict:
    """McNemar on accuracy + exact-weight sign-flip permutation on balanced accuracy (A minus B)."""
    y = [rows[i]["_agent"] for i in idx]
    ca = [pa[i] == rows[i]["_agent"] for i in idx]
    cb = [pb[i] == rows[i]["_agent"] for i in idx]
    b = sum(1 for x, z in zip(ca, cb) if x and not z)
    c = sum(1 for x, z in zip(ca, cb) if z and not x)
    ny = Counter(y)
    K = len(ny)
    d = [(int(x) - int(z)) / (K * ny[yy]) for x, z, yy in zip(ca, cb, y)]
    obs = sum(d)
    nz = [v for v in d if v]
    rng = random.Random(seed)
    hits = sum(1 for _ in range(draws) if abs(sum(v if rng.random() < 0.5 else -v for v in nz)) >= abs(obs) - 1e-12)
    return {"a_only": b, "b_only": c, "mcnemar_p": round(mcnemar_exact_p(b, c), 4), "d_ba": round(obs, 4), "perm_p_ba": round((hits + 1) / (draws + 1), 4)}


def per_project(rows, idx, preds: dict[str, list]) -> dict:
    by = defaultdict(list)
    for i in idx:
        by[rows[i]["_proj"]].append(i)
    return {p: {"n": len(ii), **{k: round(sum(v[i] == rows[i]["_agent"] for i in ii) / len(ii), 3) for k, v in preds.items()}}
            for p, ii in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m ingest.router_e3")
    p.add_argument("--outcomes", type=Path, required=True)
    p.add_argument("--exe", type=Path, required=True)
    p.add_argument("--split", choices=("tune", "held"), default="tune")
    p.add_argument("--systems", default="")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    rows = load(a.outcomes)
    mask = scored_mask(rows)
    exe = a.exe
    S = {
        "B0_majority": lambda: sys_majority(rows, mask),
        "B1_rule": lambda: sys_rule(rows, mask),
        "projmajority": lambda: sys_majority(rows, mask, scope="project"),
        "B2_live_v1": lambda: sys_mesh(rows, mask, exe, "v1"),
        "mesh_v1text": lambda: sys_mesh(rows, mask, exe, "v1text"),
        "mesh_clean": lambda: sys_mesh(rows, mask, exe, "clean"),
        "mesh_clean_verb": lambda: sys_mesh(rows, mask, exe, "clean_verb"),
        "B3_hybrid_v1": lambda: sys_mesh(rows, mask, exe, "v1", scope="perproject", prior_k=500.0, roster=True),
        "mesh_pp_clean": lambda: sys_mesh(rows, mask, exe, "clean", scope="perproject"),
        "mesh_pp_clean_roster": lambda: sys_mesh(rows, mask, exe, "clean", scope="perproject", roster=True),
        "mesh_hyb_clean": lambda: sys_mesh(rows, mask, exe, "clean", scope="perproject", prior_k=500.0, roster=True),
        "mesh_hyb_clean_k50": lambda: sys_mesh(rows, mask, exe, "clean", scope="perproject", prior_k=50.0, roster=True),
        "mesh_proj_clean": lambda: sys_mesh(rows, mask, exe, "clean", scope="project"),
    }
    # frozen E3 candidate (chosen on the tune split only, before the held-out run; see findings/exp-router.md)
    S["C_nb_frozen"] = lambda: sys_nb(rows, mask, "clean_verb", alpha=0.05, proj_w=3.0, margin=28.0, scope_cands="project", roster=True)
    for fs in ("v1", "clean", "clean_verb"):
        for pw in (0.0, 1.0, 3.0):
            for mg in (0.0, 2.0, 5.0):
                S[f"nb_{fs}_p{pw:g}_m{mg:g}"] = (lambda fs=fs, pw=pw, mg=mg: sys_nb(rows, mask, fs, proj_w=pw, margin=mg))
        S[f"nb_{fs}_projc_roster_p3_m2"] = (lambda fs=fs: sys_nb(rows, mask, fs, proj_w=3.0, margin=2.0, scope_cands="project", roster=True))
        S[f"nb_{fs}_projc_roster_p1_m0"] = (lambda fs=fs: sys_nb(rows, mask, fs, proj_w=1.0, margin=0.0, scope_cands="project", roster=True))
    names = [s for s in (a.systems.split(",") if a.systems else S) if s]
    for base in ("B0_majority", "B1_rule"):
        if base not in names:
            names.insert(0, base)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {n: ex.submit(S[n]) for n in names}
        preds = {n: f.result() for n, f in futs.items()}
    idx = _sel(rows, mask, a.split)
    fr = _sel(rows, mask, a.split, only_fr=True)
    res = {"split": a.split, "n_rows": len(idx), "n_fr": len(fr), "systems": {}}
    for n in names:
        if n == "B1_rule":
            res["systems"][n] = {"fr": metrics(rows, fr, preds[n])} if fr else {}
            continue
        e = {"all": metrics(rows, idx, preds[n]), "vs_majority": paired(rows, idx, preds[n], preds["B0_majority"])}
        if "B2_live_v1" in preds and n != "B2_live_v1":
            e["vs_live"] = paired(rows, idx, preds[n], preds["B2_live_v1"])
        if fr:
            e["fr"] = metrics(rows, fr, preds[n])
            e["fr_vs_rule"] = paired(rows, fr, preds[n], preds["B1_rule"])
        res["systems"][n] = e
    res["per_project"] = per_project(rows, idx, {n: preds[n] for n in names if n != "B1_rule"})
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n, e in res["systems"].items():
        if "all" not in e:
            continue
        al, vm = e["all"], e["vs_majority"]
        extra = f" | fr {e['fr']['acc']:.3f} vs rule p={e['fr_vs_rule']['mcnemar_p']}" if "fr" in e else ""
        print(f"{n:34s} acc {al['acc']:.3f} ba {al['ba']:.3f} | vs maj dBA {vm['d_ba']:+.3f} p={vm['perm_p_ba']} acc {vm['a_only']}/{vm['b_only']} p={vm['mcnemar_p']}{extra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
