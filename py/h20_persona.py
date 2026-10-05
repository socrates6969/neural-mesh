"""h20-A: ROLE / PERSONA-conditioned routing in seeded domain-structured worlds (SYNTHETIC simulation; no LLM involved).

ASSUMPTION (stated explicitly, it is the whole question): invoking an agent inside a role-conditioned context ("act as a <role>") changes the quality of the
answer ONLY through the model below; nothing else about the persona matters.
   q(t, agent a, role r) = clip( comp[a, d_t, s_t] + role_term + N(0, .10), 0, 1 )
   role_term = 0                      if r = generic
             = b * g_a - c            if r = d_t        (matching role: modest bonus b, agent responsiveness g_a)
             = - p - c                if r != d_t       (mismatch penalty p; c = ceremonial cost of ANY role prompt)
Effect worlds (b, p, c, g): mod (.05,.04,0,1)  zero (0,0,0,1)  harm (-.03,.06,.02,1)  strong (.12,.06,0,1)  hetero (.06,.04,0, g_a~U(-1,2.2)).
Two independent generators: G1 = specialist agents + templated task text; G2 = latent-factor agents + topic-mixture (noisier) text.  ctl = identical agents (the role is the only lever).
Roles = 10 domains {scientist, engineer, business_analyst, executive, financial_analyst, commodities_analyst, economist, linguist, psychologist, historian}.
Policies (all online, graded feedback revealed for 70% of tasks, no extra LLM calls; a role-prompted call costs +60 prompt tokens):
  fair generic baselines: eps (global agent means), pts (Gaussian Thompson per agent), kweps / kwts (per-keyword x agent table, shrunk)
  oracle_role   : role = true domain, agent by kwts
  anchor_role   : role = argmax anchor-keyword overlap (static role-descriptor anchors, 60% of the role's words + 20% foreign), agent by kwts
  learned_role  : role chosen by a keyword x role bandit from graded outcomes (11 arms), agent by kwts (factorised)
  joint         : arms = (agent, {generic, anchor role}) (2N arms) by kwts
  gate_static   : anchor role only if the anchor margin >= tau, else generic
  gate_ab       : GLOBAL online randomised A/B on anchor-eligible tasks (role vs generic, 50/50 while exploring; role is used only when the one-sided
                  lower confidence bound of the role-minus-generic difference is > 0; explore prob eps_g)
  gate_ab_agent : same, per agent (shrunk to the pooled estimate) for the hetero world
Protocol: tune seeds 1-8 (eps, gate tau/eps_g/z), held-out 31-60, confirm 61-90 (frozen).  Usage: python py/h20_persona.py --tune | --phase held|confirm
"""
import argparse, json, math, os, sys, time
import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
ROLES = ["scientist", "engineer", "business_analyst", "executive", "financial_analyst", "commodities_analyst", "economist", "linguist", "psychologist", "historian"]
R = len(ROLES)
FAM = np.array([1, 1, 0, 0, 0, 0, 0, 2, 2, 2])
NSTRONG, NSUB, NFAMW, NGEN, L = 30, 6, 20, 200, 12
A = 18
T = 1000
SIG = 0.10
REVEAL = 0.7
S_SUB = 3
OFF_STRONG = 0
OFF_SUB = R * NSTRONG
OFF_FAM = OFF_SUB + R * S_SUB * NSUB
OFF_GEN = OFF_FAM + 3 * NFAMW
V = OFF_GEN + NGEN

EFFECTS = {"mod": (0.05, 0.04, 0.0, "one"), "zero": (0.0, 0.0, 0.0, "one"), "harm": (-0.03, 0.06, 0.02, "one"),
           "strong": (0.12, 0.06, 0.0, "one"), "hetero": (0.06, 0.04, 0.0, "het")}
WORLDS = [("G1", "mod"), ("G1", "zero"), ("G1", "harm"), ("G1", "strong"), ("G1", "hetero"), ("G2", "mod"), ("G2", "zero"), ("G2", "harm"), ("ctl", "mod"), ("ctl", "zero")]


def wname(g, e):
    return "%s-%s" % (g, e)


def make_world(gen, eff, seed):
    rng = np.random.RandomState(seed * 7919 + (hash(gen) % 97 if False else {"G1": 1, "G2": 2, "ctl": 3}[gen]) * 104729)
    b, p, c, gk = EFFECTS[eff]
    # ---- agents
    comp = np.zeros((A, R, S_SUB))
    if gen == "G2":
        fc = rng.normal(0, 1, (3, 3))
        v = np.stack([fc[FAM[d]] + rng.normal(0, 0.5, 3) for d in range(R)])
        u = rng.normal(0, 1, (A, 3))
        z = u @ v.T
        z = (z - z.mean()) / z.std()
        base = np.clip(0.55 + 0.15 * z + rng.normal(0, 0.03, z.shape), 0.1, 0.95)
    elif gen == "ctl":
        base = np.full((A, R), 0.60)
    else:
        base = np.zeros((A, R))
        kinds = ["spec"] * 12 + ["gen"] * 3 + ["weak"] * 3
        for a in range(A):
            if kinds[a] == "spec":
                d0 = rng.randint(R)
                same = [d for d in range(R) if FAM[d] == FAM[d0] and d != d0]
                d1 = rng.choice(same) if (same and rng.rand() < 0.5) else rng.choice([d for d in range(R) if d != d0])
                for d in range(R):
                    if d in (d0, d1):
                        base[a, d] = rng.uniform(0.78, 0.92)
                    elif FAM[d] == FAM[d0] or FAM[d] == FAM[d1]:
                        base[a, d] = rng.uniform(0.50, 0.65)
                    else:
                        base[a, d] = rng.uniform(0.30, 0.50)
            elif kinds[a] == "gen":
                base[a] = rng.uniform(0.52, 0.64, R)
            else:
                base[a] = rng.uniform(0.30, 0.45, R)
    comp = np.clip(base[:, :, None] + rng.normal(0, 0.04, (A, R, S_SUB)) * (0 if gen == "ctl" else 1), 0.05, 0.98)
    g = np.ones(A) if gk == "one" else rng.uniform(-1.0, 2.2, A)
    # ---- tasks
    dm = rng.choice(R, T, p=np.full(R, 1.0 / R))
    sub = rng.randint(S_SUB, size=T)
    tok = np.zeros((T, L), dtype=np.int64)
    for t in range(T):
        d, s = dm[t], sub[t]
        if gen == "G2":
            sibs = [x for x in range(R) if FAM[x] == FAM[d] and x != d] or [d]
            for k in range(L):
                u = rng.rand()
                if u < 0.35:
                    tok[t, k] = OFF_STRONG + d * NSTRONG + min(int(rng.exponential(8)), NSTRONG - 1)
                elif u < 0.50:
                    tok[t, k] = OFF_FAM + FAM[d] * NFAMW + rng.randint(NFAMW)
                elif u < 0.62:
                    tok[t, k] = OFF_STRONG + rng.choice(sibs) * NSTRONG + min(int(rng.exponential(8)), NSTRONG - 1)
                elif u < 0.70:
                    tok[t, k] = OFF_STRONG + rng.randint(R) * NSTRONG + min(int(rng.exponential(8)), NSTRONG - 1)
                elif u < 0.78:
                    tok[t, k] = OFF_SUB + (d * S_SUB + s) * NSUB + rng.randint(NSUB)
                else:
                    tok[t, k] = OFF_GEN + rng.randint(NGEN)
        else:
            sibs = [x for x in range(R) if FAM[x] == FAM[d] and x != d] or [d]
            row = [OFF_STRONG + d * NSTRONG + x for x in rng.choice(NSTRONG, 3, replace=False)]
            row += [OFF_SUB + (d * S_SUB + s) * NSUB + x for x in rng.choice(NSUB, 2, replace=False)]
            row += [OFF_FAM + FAM[d] * NFAMW + x for x in rng.choice(NFAMW, 2, replace=False)]
            row += [OFF_STRONG + rng.choice(sibs) * NSTRONG + rng.randint(NSTRONG)] if rng.rand() < 0.6 else [OFF_GEN + rng.randint(NGEN)]
            while len(row) < L:
                row.append(OFF_GEN + rng.randint(NGEN))
            tok[t] = np.array(row)
    # ---- anchors: role descriptor words = 60% of the role's strong words + 20% (6 words) foreign (random other-role strong words)
    anc = np.zeros((R, V), dtype=np.int8)
    for r in range(R):
        own = rng.choice(NSTRONG, int(0.6 * NSTRONG), replace=False)
        anc[r, OFF_STRONG + r * NSTRONG + own] = 1
        for _ in range(6):
            o = rng.choice([x for x in range(R) if x != r])
            anc[r, OFF_STRONG + o * NSTRONG + rng.randint(NSTRONG)] = 1
    cnt = anc[:, tok].sum(2).T.astype(float) + rng.rand(T, R) * 0.01  # (T,R), random tie break
    order = np.argsort(-cnt, 1)
    pred = order[:, 0]
    margin = cnt[np.arange(T), order[:, 0]] - cnt[np.arange(T), order[:, 1]]
    noise = rng.normal(0, SIG, (T, A))
    reveal = rng.rand(T) < REVEAL
    fbn = rng.normal(0, 0.05, T)
    return dict(b=b, p=p, c=c, g=g, comp=comp, dm=dm, sub=sub, tok=tok, pred=pred, margin=margin, noise=noise, reveal=reveal, fbn=fbn, rng_seed=seed)


def role_term(w, a, role, t):
    """role: -1 generic, else role index"""
    if role < 0:
        return 0.0
    if role == w["dm"][t]:
        return w["b"] * w["g"][a] - w["c"]
    return -w["p"] - w["c"]


class KW:
    """keyword x arm table (shrunk), greedy / Thompson choice"""

    def __init__(self, narms, rng, k=2.0, ts=True, sig=0.15):
        self.S = np.zeros((V, narms))
        self.C = np.zeros((V, narms))
        self.gs = np.zeros(narms)
        self.gc = np.zeros(narms)
        self.k, self.ts, self.sig, self.rng = k, ts, sig, rng

    def est(self, toks):
        gm = (self.gs + 0.5) / (self.gc + 1.0)
        s = self.S[toks]
        c = self.C[toks]
        e = ((s + self.k * gm) / (c + self.k)).mean(0)
        return e, c.mean(0)

    def choose(self, toks, eps=0.0):
        if eps > 0 and self.rng.rand() < eps:
            return int(self.rng.randint(len(self.gs)))
        e, ne = self.est(toks)
        if self.ts:
            e = e + self.sig * self.rng.randn(len(e)) / np.sqrt(1.0 + ne)
        return int(np.argmax(e))

    def update(self, toks, arm, q):
        self.S[toks, arm] += q
        self.C[toks, arm] += 1
        self.gs[arm] += q
        self.gc[arm] += 1


def run(w, kind, prm, seed):
    """returns per-task quality, fraction of calls that used a role prompt"""
    rng = np.random.RandomState(seed * 31 + 5)      # AGENT-selection stream: identical across all policies that share the kweps chooser
    rng_g = np.random.RandomState(seed * 37 + 11)   # role-exploration stream (separate, so a zero-effect role cannot change agent choices)
    Tn = T
    q = np.zeros(Tn)
    used = 0
    tok, dm, comp, noise = w["tok"], w["dm"], w["comp"], w["noise"]
    sub = w["sub"]

    def outcome(t, a, role):
        return float(np.clip(comp[a, dm[t], sub[t]] + role_term(w, a, role, t) + noise[t, a], 0, 1))

    if kind in ("eps", "pts"):
        sm = np.zeros(A)
        n = np.zeros(A)
        for t in range(Tn):
            if kind == "eps":
                a = int(rng.randint(A)) if rng.rand() < prm["eps"] else int(np.argmax((sm + 0.5) / (n + 1)))
            else:
                a = int(np.argmax((sm + 0.5) / (n + 1) + 0.15 * rng.randn(A) / np.sqrt(n + 1)))
            q[t] = outcome(t, a, -1)
            if w["reveal"][t]:
                f = q[t] + w["fbn"][t]
                sm[a] += f
                n[a] += 1
        return q, 0.0
    if kind in ("kweps", "kwts"):
        kw = KW(A, rng, ts=(kind == "kwts"))
        for t in range(Tn):
            a = kw.choose(tok[t], prm.get("eps", 0.1) if kind == "kweps" else 0.0)
            q[t] = outcome(t, a, -1)
            if w["reveal"][t]:
                kw.update(tok[t], a, q[t] + w["fbn"][t])
        return q, 0.0
    # ---- treatments: agent selection by kwts unless stated
    ts = prm.get("kwts_agent", True)
    if kind == "joint":
        kw = KW(2 * A, rng, ts=False)
        for t in range(Tn):
            arm = kw.choose(tok[t], prm.get("eps_agent", 0.1))
            a, mode = arm % A, arm // A
            role = w["pred"][t] if mode == 1 else -1
            q[t] = outcome(t, a, role)
            used += mode
            if w["reveal"][t]:
                kw.update(tok[t], arm, q[t] + w["fbn"][t])
        return q, used / Tn
    kwa = KW(A, rng, ts=False)  # treatments use the SAME agent chooser as the best fair keyword baseline (kweps, eps tuned) -> contrasts isolate the role effect
    kwr = KW(R + 1, rng_g, ts=False) if kind == "learned_role" else None
    # A/B state
    ab_n = np.zeros((A + 1, 2))   # [agent or global idx][0=generic,1=role]
    ab_s = np.zeros((A + 1, 2))
    ab_ss = np.zeros((A + 1, 2))
    for t in range(Tn):
        a = kwa.choose(tok[t], prm.get("eps_agent", 0.1))
        e_pre = kwa.est(tok[t])[0][a]
        explore_flag = False
        if kind == "oracle_role":
            role = dm[t]
        elif kind == "anchor_role":
            role = w["pred"][t]
        elif kind == "learned_role":
            ra = kwr.choose(tok[t], prm.get("eps_agent", 0.1))
            role = ra - 1  # arm 0 = generic
        elif kind == "gate_static":
            role = w["pred"][t] if w["margin"][t] >= prm["tau"] else -1
        else:  # gate_ab / gate_ab_agent
            role = -1
            if w["margin"][t] >= prm["tau"]:
                gi = a if kind == "gate_ab_agent" else A
                use = False
                # pooled estimate (shrinkage target)
                def diff(i):
                    n0, n1 = ab_n[i]
                    if n0 < 2 or n1 < 2:
                        return None, None, n0 + n1
                    m0, m1 = ab_s[i, 0] / n0, ab_s[i, 1] / n1
                    v0 = max(ab_ss[i, 0] / n0 - m0 * m0, 1e-4)
                    v1 = max(ab_ss[i, 1] / n1 - m1 * m1, 1e-4)
                    return m1 - m0, math.sqrt(v0 / n0 + v1 / n1), n0 + n1
                if rng_g.rand() < prm["eps_g"]:
                    explore_flag = True
                    role = w["pred"][t] if rng_g.rand() < 0.5 else -1
                else:
                    d_, se_, n_ = diff(gi)
                    if kind == "gate_ab_agent":
                        dg, seg, ng = diff(A)
                        if d_ is None or n_ < 12:
                            d_, se_ = dg, seg  # fall back to pooled estimate until the agent has its own
                    if d_ is not None and d_ - prm["z"] * se_ > 0:
                        role = w["pred"][t]
        q[t] = outcome(t, a, role)
        used += (role >= 0)
        if w["reveal"][t]:
            f = q[t] + w["fbn"][t]
            kwa.update(tok[t], a, f)
            if kwr is not None:
                kwr.update(tok[t], role + 1, f)
            if explore_flag:
                m = 1 if role >= 0 else 0
                r_ = f - e_pre  # residual vs the agent's current keyword estimate removes task-mix heterogeneity (assignment is randomised)
                for gi in (A, a):
                    ab_n[gi, m] += 1
                    ab_s[gi, m] += r_
                    ab_ss[gi, m] += r_ * r_
    return q, used / Tn


def ceilings(w):
    """analytic: mean over tasks of the best agent's expected quality, without / with the oracle role"""
    d, s = w["dm"], w["sub"]
    cm = w["comp"][:, d, s]  # (A,T)
    best_gen = cm.max(0).mean()
    rt = w["b"] * w["g"][:, None] - w["c"]
    best_role = (cm + rt).max(0).mean()
    return float(best_gen), float(best_role)


POL = {
    "eps": ("eps", {"eps": 0.1}), "pts": ("pts", {}), "kweps": ("kweps", {"eps": 0.1}), "kwts": ("kwts", {}),
    "oracle_role": ("oracle_role", {}), "anchor_role": ("anchor_role", {}), "learned_role": ("learned_role", {}), "joint": ("joint", {}),
    "gate_static": ("gate_static", {"tau": 2}), "gate_ab": ("gate_ab", {"tau": 2, "eps_g": 0.15, "z": 1.28}), "gate_ab_agent": ("gate_ab_agent", {"tau": 2, "eps_g": 0.15, "z": 1.28}),
}
BASE = ["eps", "pts", "kweps", "kwts"]
TREAT = ["oracle_role", "anchor_role", "learned_role", "joint", "gate_static", "gate_ab", "gate_ab_agent"]


def run_world(gen, eff, seed, pol):
    w = make_world(gen, eff, seed)
    out = {}
    for name, (kind, prm) in pol.items():
        q, used = run(w, kind, prm, seed)
        out[name] = {"h": float(q.mean()), "late": float(q[T // 2:].mean()), "early": float(q[:T // 4].mean()), "role_frac": used}
    out["_ceil"] = ceilings(w)
    out["_anchor_acc"] = float(np.mean(w["pred"] == w["dm"]))
    return out


def paired(a, b, tost=None):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m / max(se, 1e-12)), n - 1)
    o = {"diff": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "p": float(p), "n": n}
    if tost is not None:
        pl = stats.t.sf((m + tost) / max(se, 1e-12), n - 1)
        pu = stats.t.cdf((m - tost) / max(se, 1e-12), n - 1)
        o["tost_p"] = float(max(pl, pu))
        o["equiv"] = bool(o["tost_p"] < 0.05)
    return o


def holm(ps, alpha=0.05):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    ok = [False] * len(ps)
    for rank, i in enumerate(idx):
        if ps[i] <= alpha / (len(ps) - rank):
            ok[i] = True
        else:
            break
    return ok


def job(args):
    gen, eff, seed, pol = args
    return (gen, eff, seed, run_world(gen, eff, seed, pol))


def seeds_of(phase):
    return {"tune": range(1, 9), "held": range(31, 61), "confirm": range(61, 91)}[phase]


def run_phase(phase, pol, worlds=WORLDS, procs=2):
    from multiprocessing import Pool
    jobs = [(g, e, s, dict(pol)) for (g, e) in worlds for s in seeds_of(phase)]
    with Pool(procs) as P:
        res = P.map(job, jobs, chunksize=4)
    R_ = {}
    for g, e, s, o in res:
        R_.setdefault(wname(g, e), {})[s] = o
    return R_


def summarize(R_, phase, pol):
    out = {"phase": phase, "worlds": {}}
    allp, keys = [], []
    for wn, runs in R_.items():
        seeds = sorted(runs)
        arr = lambda p_, k="h": np.array([runs[s][p_][k] for s in seeds])
        bestb = max(BASE, key=lambda b: arr(b).mean())  # best fair baseline picked on the evaluated seeds (conservative)
        W = {"best_fair_baseline": bestb, "pol": {}, "contrasts": {}, "ceil_generic_agent": float(np.mean([runs[s]["_ceil"][0] for s in seeds])),
             "ceil_agent_plus_oracle_role": float(np.mean([runs[s]["_ceil"][1] for s in seeds])), "anchor_acc": float(np.mean([runs[s]["_anchor_acc"] for s in seeds]))}
        for p_ in pol:
            W["pol"][p_] = {"h": float(arr(p_).mean()), "late": float(arr(p_, "late").mean()), "early": float(arr(p_, "early").mean()), "role_frac": float(arr(p_, "role_frac").mean())}
        for p_ in TREAT:
            if p_ not in pol:
                continue
            for ref_name, ref in (("bestfair", bestb), ("kwts", "kwts")):
                for k in ("h", "late", "early"):
                    c = paired(arr(p_, k), arr(ref, k), tost=0.01)
                    W["contrasts"]["%s_vs_%s_%s" % (p_, ref_name, k)] = c
                    if ref_name == "bestfair" and k == "h":
                        allp.append(c["p"])
                        keys.append((wn, p_))
        out["worlds"][wn] = W
    ok = holm(allp)
    out["holm"] = [{"world": k[0], "policy": k[1], "p": p, "holm_sig": o} for k, p, o in zip(keys, allp, ok)]
    sig = {(h["world"], h["policy"]): h["holm_sig"] for h in out["holm"]}
    for wn, W in out["worlds"].items():
        for p_ in TREAT:
            c = W["contrasts"].get("%s_vs_bestfair_h" % p_)
            if c:
                c["holm_sig"] = sig[(wn, p_)]
    return out


def print_summary(S):
    print("== phase", S["phase"])
    for wn, W in S["worlds"].items():
        print("-- %-10s anchor_acc %.2f | ceilings: best agent %.3f, +oracle role %.3f | best fair baseline: %s" % (wn, W["anchor_acc"], W["ceil_generic_agent"], W["ceil_agent_plus_oracle_role"], W["best_fair_baseline"]))
        for p_, m in W["pol"].items():
            line = "   %-14s H %.4f late %.4f early %.4f role_calls %.2f" % (p_, m["h"], m["late"], m["early"], m["role_frac"])
            c = W["contrasts"].get("%s_vs_bestfair_h" % p_)
            if c:
                line += "  vs bestfair %+.4f [%+.4f,%+.4f]%s%s" % (c["diff"], c["lo"], c["hi"], " HOLM" if c.get("holm_sig") else "", ("  TOST.01 %s" % c["equiv"]))
            print(line)


def tune(procs=2):
    # stage 1: baseline eps on seeds 1-8.  stage 2: gate (tau, eps_g, z) with the tuned agent eps; objective = mean over worlds of (gate - kweps) H
    # plus the worst-case world (guard against harm).
    grid_eps = [0.02, 0.05, 0.1, 0.2]
    pol = {}
    for e in grid_eps:
        pol["eps%.2f" % e] = ("eps", {"eps": e})
        pol["kweps%.2f" % e] = ("kweps", {"eps": e})
    pol["kwts"] = POL["kwts"]
    pol["pts"] = POL["pts"]
    POL.update(pol)
    R1 = run_phase("tune", pol, procs=procs)
    eps_agent = float(max(grid_eps, key=lambda e: np.mean([np.mean([runs[x]["kweps%.2f" % e]["h"] for x in runs]) for runs in R1.values()])))
    gates = [(tau, eg, z) for tau in (0, 1, 2, 3) for eg in (0.1, 0.2, 0.3) for z in (0.0, 0.5, 1.28)]
    pol2 = {"kweps_t": ("kweps", {"eps": eps_agent})}
    for tau, eg, z in gates:
        pol2["gab|%d|%.1f|%.2f" % (tau, eg, z)] = ("gate_ab", {"tau": tau, "eps_g": eg, "z": z, "eps_agent": eps_agent})
        pol2["gabA|%d|%.1f|%.2f" % (tau, eg, z)] = ("gate_ab_agent", {"tau": tau, "eps_g": eg, "z": z, "eps_agent": eps_agent})
    for tau in (0, 1, 2, 3, 4):
        pol2["gs|%d" % tau] = ("gate_static", {"tau": tau, "eps_agent": eps_agent})
    POL.update(pol2)
    R_ = run_phase("tune", pol2, procs=procs)
    pol = pol2
    rows = []
    for name in pol:
        if not name.startswith(("gab", "gs")):
            continue
        diffs = {}
        for wn, runs in R_.items():
            s = sorted(runs)
            diffs[wn] = float(np.mean([runs[x][name]["h"] - runs[x]["kweps_t"]["h"] for x in s]))
        rows.append((float(np.mean(list(diffs.values()))), min(diffs.values()), name, diffs))
    rows.sort(key=lambda r: (r[0] + r[1]), reverse=True)  # mean + worst-case world
    eps_pick = {"kweps": eps_agent}
    eps_pick["eps"] = float(max(grid_eps, key=lambda e: np.mean([np.mean([runs[x]["eps%.2f" % e]["h"] for x in runs]) for runs in R1.values()])))
    best_ab = [r for r in rows if r[2].startswith("gab|")][0]
    best_aba = [r for r in rows if r[2].startswith("gabA|")][0]
    best_gs = [r for r in rows if r[2].startswith("gs|")][0]
    tuned = {"eps": eps_pick["eps"], "kweps": eps_pick["kweps"], "gate_ab": best_ab[2], "gate_ab_agent": best_aba[2], "gate_static": best_gs[2],
             "table": [(r[2], round(r[0], 4), round(r[1], 4)) for r in rows[:14]]}
    json.dump(tuned, open(os.path.join(RES, "h20_persona_tune.json"), "w"), indent=1)
    print(json.dumps(tuned, indent=1))
    return tuned


def frozen_pol(tuned):
    def parse(n):
        _, tau, eg, z = n.split("|")
        return {"tau": int(tau), "eps_g": float(eg), "z": float(z)}
    pol = dict(POL)
    pol["eps"] = ("eps", {"eps": tuned["eps"]})
    pol["kweps"] = ("kweps", {"eps": tuned["kweps"]})
    ea = {"eps_agent": tuned["kweps"]}
    pol["gate_ab"] = ("gate_ab", {**parse(tuned["gate_ab"]), **ea})
    pol["gate_ab_agent"] = ("gate_ab_agent", {**parse(tuned["gate_ab_agent"]), **ea})
    pol["gate_static"] = ("gate_static", {"tau": int(tuned["gate_static"].split("|")[1]), **ea})
    for k in ("oracle_role", "anchor_role", "learned_role", "joint"):
        pol[k] = (pol[k][0], {**pol[k][1], **ea})
    return pol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--worlds", default=None)
    a = ap.parse_args()
    if a.tune:
        tune(a.procs)
        return
    tuned = json.load(open(os.path.join(RES, "h20_persona_tune.json")))
    pol = frozen_pol(tuned)
    POL.update(pol)
    names = BASE + TREAT
    R_ = run_phase(a.phase, {n: pol[n] for n in names}, procs=a.procs)
    S = summarize(R_, a.phase, names)
    S["tuned"] = {k: v for k, v in tuned.items() if k != "table"}
    print_summary(S)
    json.dump(S, open(os.path.join(RES, "h20_persona_%s.json" % a.phase), "w"), indent=1)


if __name__ == "__main__":
    main()
