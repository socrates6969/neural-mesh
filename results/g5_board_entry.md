# G5 (W5 DESIGN-BIZ) board entry: design / business / marketing task world, hierarchical routing + verified retrieval/reuse

Files (all mine, src/*.rs untouched, no commits, no new crates/models): `src/bin/g5_study.rs`, `src/bin/g5_common/mod.rs` (world, routers, driver, 4 unit tests),
`results/g5_study.{json,log}` (held-out 31-60), `results/g5_study_confirm61-90.{json,log}` (same frozen tuning), `results/g5_sens.{json,log}` and `results/g5_sens_confirm61-90.{json,log}`
(one-at-a-time sensitivity), `results/g5_check.log`, `results/g5_world_examples.txt`, `results/g5_proposed_patch.md`.

HONESTY: everything is a seeded simulation I wrote. It measures routing a task to the right specialist and supplying a verified recurring brief/style guide to that
specialist. It does not measure model creativity or any real LLM behaviour. The retrieval gain exists because I made task quality depend on having the right brief
(parameter d); the interesting questions are how much of that dependence the pipeline realises, where it breaks, and whether controls regress. No "10x", no novel-knowledge claim.

## The world (src/bin/g5_common/mod.rs; `g5_study --show` prints agents, correlations and sample tasks -> results/g5_world_examples.txt)
- 5 domains x 4 sub-skills: design (ux-flow, visual-ui, design-system, accessibility), business (financial-model, go-to-market, ops-plan, pricing), marketing (copywriting, seo, campaign, social),
  coding and math (controls). Task = tags (`task_type = "domain/sub"`, feature `dom:<domain>`) + text tokens (domain word, 2 of 3 sub-skill words, optional cross-talk word, 2 filler words, 3 project tokens
  when the task belongs to a recurring project). 1500 tasks per run, domain mix e.g. design .30 / business .20 / marketing .25 / coding .15 / math .10.
- Agents (hidden competence per sub-skill, then shuffled): Agency pool = 10 agents (2 generalists ~.5, 5 domain specialists ~.78-.90 in-domain / ~.35 off, 1 "creative" agent strong in design AND marketing
  (cross-domain correlation), 1 accessibility expert, 1 financial-model expert). Other pools: Generalists (8, noisy mild differences), Big (20: 3 specialists per domain + 5 generalists), Latent
  (10 agents whose competence comes from 3 latent factors creative/analytic/technical: measured design~marketing agent correlation .74 in the shown seed), Equal (8 identical agents).
- Quality (graded, in [0,1], common random numbers across arms): q = clamp(competence + task difficulty + agent noise) x fit. fit = 1 - d(1 - neff*u): d = context dependence of the domain (design .30, marketing .28,
  business .22, coding 0, math 0); neff = note strength x (2*purity - 1) where purity = the fraction of records behind the retrieved anchor that really belong to the SAME project (ground truth held by the executor);
  a wrong brief is worse than none (extra penalty mis = .5); u = .5 + .5*competence (stronger agents use a brief better). So retrieval can both help and hurt, and its quality is set by what the hash embedding retrieves.
- Feedback to EVERY arm (baselines included): graded quality + noise, revealed for 60% of tasks, 15 tasks late. Baselines are therefore not handicapped with the raw hook bit (that was e3's `plain`).
- Keyword map: sub-skill words advertise the best agent, 30% of sub-skills advertise a random agent (decoy). Retrieval = `ResearchSession` store with the default 64-d hash embedding (the MiniLM fixture only covers the f5 corpus text, so it cannot embed these tasks).
- Worlds. Treatment (context matters): agency (main), agency-briefs (briefs ingested up front via the approved-source `FixtureFetcher` path), generalist-pool, big-pool (20 agents, business-heavy mix, 24 projects),
  latent-factors, weak-context (d x .4), low-recurrence (200 projects over 1500 tasks). Stress: confusable-briefs (projects differ by 1 token among shared brand/style tokens). Controls (d = 0): ctl-nocontext-design (design-heavy, no recurring tokens),
  ctl-coding-math (coding+math only, recurring tokens present but useless), ctl-equal-agents.

## Arms
Fair baselines (9, same delayed graded feedback, no retrieval): keyword, eps-greedy (tuned), pooled-thompson (tuned), hebbian-robust, warm(hebr) (k tuned = 1), eps(.9,0)[e1], pts(1)[e1], warm(k=1)[e1], guarded-safe = `defaults::recommended_router` ("plain").
Treatments: `hier` (my 3-level shrinkage router agent -> domain -> sub-skill with keyword pseudo-count, Thompson draw; no retrieval); `session(warm+retr)` and `session(safe+retr)` (the real `research::ResearchSession`, retrieval + reuse, critic OFF as f7 recommends,
router = WarmHebr / GuardedHebr(recommended_guard)); `hier+retr` (in-bin replica of `run_task` around `hier`, see check below). Reference rows: oracle(no-ctx) = best agent per task without a brief; oracle(+perfect ctx); ablation `hier(no domain level)`.
Tuned on seeds 1-8 only (objective = mean of H250/H500/H1500 over all 11 worlds): eps(1,0), pts gamma 1, warm k=1, hier m_g 2 m_d 2 m_s 6 xi .2 kw_w 8, retrieval store_min .55 sim_min .75 (grid store {.55,.65,.75} x sim {.55,.65,.75,.85}; .75 is interior). The library defaults (store .6, sim .6) are NOT what is used.
Best fair baseline = best mean among the 9 fair arms, chosen per world x horizon on the held-out data (conservative: selection works against the treatment).

### VERIFY REQUEST — W5 — G5-A: replica == real ResearchSession, and d=0 controls are a bit-identical no-op for retrieval
rerun: `export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r5 RAYON_NUM_THREADS=3; cd /c/Users/mariu/neural-mesh; cargo run --release --offline --bin g5_study -- --check` (~3 s); unit tests: `cargo test --release --offline --bin g5_study` (4 pass).
claim: for 6 worlds x {guarded-safe, warm k=3} x 3 seeds, per-task delivered quality of `run_replica` (retrieval off and on, incl. the briefs-ingest world) equals `ResearchSession` (health off) with max |dq| = 0.0; in the three d=0 worlds retrieval on == off bit-identically for safe, warm and hier. This is why `hier+retr` (not a library path) is a fair stand-in and why the controls cannot show a retrieval regression by construction (see negatives).

### VERIFY REQUEST — W5 — G5-B: does verified retrieval/reuse + hierarchical routing raise graded quality in design/business/marketing? (main result)
rerun (deterministic; two runs identical modulo `runtime_s`, diffed for the 61-90 file): same exports as above, then
  `cargo run --release --offline --bin g5_study -- --out results/g5_study.json` (~50 s incl. tuning on seeds 1-8; held-out 31-60)
  `cargo run --release --offline --bin g5_study -- --held 61-90 --out results/g5_study_confirm61-90.json` (~50 s, SAME frozen tuning, fresh seeds)
Protocol: paired 95% t-CI over 30 seeds, Holm per family (F1 treatments x {best fair, plain, keyword} x worlds x H100/250/500/1500 = 528 tests; F2 ablations 308; F3 per-domain vs best fair 200; F4 sample efficiency 132).
Holm WIN/LOSS counts: F1 317W/10L (31-60), 297W/15L (61-90); F2 168W/0L, 158W/1L; F3 68W/8L, 67W/5L; F4 11W/2L, 23W/0L.

H1500 mean delivered quality (31-60), then diff vs BEST FAIR baseline, paired [95% CI] (31-60 | 61-90 point; Holm verdict W/L/ns same order):
```
world                 best-fair  hier              session(warm+retr)  session(safe+retr)  hier+retr           oracle(noctx) oracle(+ctx)
agency                .653 warm   +.006 [.004,.009]  +.101 [.097,.105]   +.089 [.081,.096]   +.107 [.102,.111]   .678          .841
                                  W|ns(+.007)        W|W (+.102)         W|W (+.090)         W|W (+.109)
agency-briefs         .645 warm   +.007 ns|+.011 W   +.102 W|+.105 W     +.080 W|+.095 W     +.111 W|+.116 W     .676          .840
generalist-pool       .546 warm   +.009 W|+.007 W    +.067 W|+.071 W     +.068 W|+.067 W     +.081 W|+.082 W     .572          .705
big-pool              .653 warm   +.014 W|+.010 ns   +.046 W|+.046 W     +.044 W|+.046 W     +.060 W|+.059 W     .695          .846
latent-factors        .663 warm   +.014 W|+.011 W    +.111 W|+.114 W     +.102 W|+.106 W     +.126 W|+.127 W     .687          .857
weak-context (d x.4)  .747 warm   +.010 ns|+.005 ns  +.044 W|+.039 W     +.028 W|+.025 W     +.053 W|+.045 W     .784          .849
low-recurrence        .645 warm   +.011 W|+.011 W    +.001 ns|+.000 ns   -.018 L|-.016 L     +.013 W|+.012 W     .679          .843
confusable (stress)   .628 warm   +.029 W|+.031 W    +.005 ns|+.007 ns   +.005 ns|+.006 W    +.037 W|+.040 W     .677          .841
ctl-nocontext-design  .831 warm   +.013 ns|+.006 ns  +.000 (identical)   -.019 L|-.016 ns    +.013 ns|+.006 ns   .869          .869
ctl-coding-math       .830 warm   +.000 ns|-.000 ns  +.000 (identical)   -.017 L|-.023 L     +.000 ns|-.000 ns   .835          .835
ctl-equal-agents      .700        -.000 ns|+.000 ns  all four within +-.0003, ns
```
(the 61-90 "best fair" for confusable is guarded-safe; mean levels there agree with 31-60 within .01.) Independent treatment worlds with a Holm WIN vs best fair at H1500 (31-60 | 61-90): session(warm+retr) 6/7 | 6/7; session(safe+retr) 6/7 | 6/7; hier+retr 7/7 | 7/7; hier alone 5/7 | 4/7.
At H500: session(warm+retr) 6/7 | 6/7, hier+retr 6/7 | 6/7, hier 2/7 | 0/7. Worlds vary pool structure (Agency / Generalists / Big 20 agents / Latent factors), domain mix, project recurrence (12 / 24 / 200), context size.
What `hier+retr` realises: agency .760 vs oracle(no-ctx) .678 and oracle(+perfect brief) .841 -> ~50% of the headroom between "perfect routing, no brief" and "perfect routing, perfect brief".

ATTRIBUTION (F2, H1500, 31-60 | 61-90): retrieval on warm = +.101/+.102 (agency), +.102/+.105 (briefs), +.067/+.071, +.046/+.046, +.111/+.114, +.044/+.039 (weak), +.001/+.000 (low-recurrence, ns), +.005/+.008 (confusable); same size on guarded-safe and on hier.
hier vs tuned warm (no retrieval): +.006..+.014 where significant (see table). Domain level inside hier: `hier` minus `hier(no domain level)` = +.027/+.026 (agency), +.005/+.006 (generalists), +.028/+.028 (big), +.010/+.009 (latent), +.022/+.025 (weak), +.028/+.027 (low-rec), Holm WIN in every world except ctl-equal-agents (also +.015/+.010 no-context design, +.004/+.005 coding-math), both seed sets -> the HIERARCHY (not just a shrinkage bandit) is what helps hier; but hier is only marginally above warm(hebr).
hier+retr minus session(warm+retr): +.014 (generalist), +.013 (big), +.016 (latent), +.012 (low-rec), +.032 (confusable) Holm WIN; ns in agency/briefs/weak.

PER-DOMAIN (F3; H1500 means warm -> hier+retr, 31-60; paired CI vs best fair per domain all Holm WIN in the context domains):
  agency: design .566 -> .724 (+.158 [.144,.172] vs hebbian-robust), business .636 -> .765, marketing .589 -> .725; coding .836 -> .837, math .829 -> .826 (controls unchanged).
  latent-factors: design .594 -> .767, business .668 -> .816, marketing .603 -> .760; coding .812 -> .840, math .801 -> .811. generalist-pool: design .507 -> .617, business .502 -> .579, marketing .540 -> .647, coding .637 -> .662.
  big-pool design .551 -> .675, business .637 -> .671, marketing .574 -> .689. Losses: design in low-recurrence (session warm+retr -.011 / -.013) and in confusable (-.025 31-60 only) = wrong-brief harm.

SAMPLE EFFICIENCY (F4): tasks until the trailing-100 mean reaches 95% of the best fair baseline's OWN late (last-300) quality (mean over 30 seeds; 100 is the minimum possible; 1501 = never; 31-60):
  agency: warm 146, guarded-safe 218, keyword 665, hier 172, session(warm+retr) 115, hier+retr 123; agency-briefs: 175 / 296 / 839 / 205 / 122 / 135; generalist-pool 139 / 136 / 209 / 136 / 118 / 107; big-pool 179 / 211 / 507 / 153 / 154 / 137; latent 123 / 203 / 618 / 111 / 109 / 105; weak 181 / 349 / 1197 / 187 / 142 / 150; low-rec 174 / 307 / 637 / 194 / 175 / 197; confusable 256 / 220 / 422 / 142 / 221 / 136.
  Holm-significant (t@95%bf): agency-briefs session(warm+retr) -53 tasks, hier+retr -41 (both sets for session: -52); generalist-pool session -21 (31-60), -42 session / -50 hier+retr (61-90); big-pool -28 / -44 (61-90); confusable hier+retr -121 (31-60). Elsewhere ns: the headline is quality LEVEL, not speed. `hier` alone is NOT faster than warm (agency 172 vs 146; first-100-task quality -.010..-.020 vs best fair in 6 of 8 treatment/stress worlds, Holm-sig only big-pool 61-90).
  Oracle-based targets (t90/t95 of oracle(no-ctx)) are in the JSON (`t90_mean`, `t95_mean`); retrieval arms exceed oracle(no-ctx) because the oracle has no brief, so those are not pure learning-speed numbers.

### VERIFY REQUEST — W5 — G5-C: sensitivity of the retrieval gain to the assumptions I built in (and where it breaks)
rerun: `cargo run --release --offline --bin g5_study -- --sens --out results/g5_sens.json` and `... --sens --held 61-90 --out results/g5_sens_confirm61-90.json` (~40 s each). One-at-a-time from `agency`, H1500, paired diff of retrieval vs the same router without it (31-60; raw CIs, no multiplicity correction; 61-90 file agrees within ~.01):
```
variant                          retr|warm   retr|hier   ctx-hit  purity
base                             +.101       +.101       .89      .98
wrong-brief penalty mis 0 / 1    +.099 / +.101  +.099 / +.103   -    .98
context size d x .25 / .5 / 1.5  +.027 / +.052 / +.109  +.026 / +.051 / +.111   (retrieval gain is ~ linear in the d I assume)
project id tokens 3/3, 2/3, 1/3  +.101 / +.046 / -.008   +.101 / +.049 / -.006   purity .98 / .79 / .47
projects 4 / 6 / 12 / 24 / 60 / 200   +.116 / +.108 / +.101 / +.085 / +.025 / +.001   (rare recurrence -> nothing to retrieve)
keyword decoy 0 / .3 / .6        +.102 / +.101 / +.097 (warm); safe +.113 / +.097 / +.080
context used equally (u=1) / only by strong agents (u=c^2)   +.110 / +.072   (agent-dependence of context use matters ~.04)
briefs ingested up front         +.100;   ... AND 1/3 id tokens: -.009 (warm), -.007 (hier)
hier minus warm (no retrieval)   +.006 base; +.002..+.015 across variants; ns at d x.25 and x.5 (the hier effect does not depend on d)
```
Reading: the pipeline's gain is real only when (a) the brief matters (d), (b) the project is identifiable from the task text, (c) projects recur. Under weak identification it is slightly NEGATIVE (-.007..-.009) because the hash embedding retrieves briefs of the wrong project (purity .46-.47).

## NEGATIVES / LIMITS (read these first)
1. The retrieval headline (+.04..+.11) is the dependence on a brief that I wrote into the world; sensitivity shows it scales ~linearly with d and vanishes with weak project identification or rare recurrence. It says nothing about real LLM outputs or real briefs. Hash embedding only (the MiniLM fixture cannot embed synthetic task text).
2. Hierarchical routing alone is a small effect: +.005..+.014 over tuned warm(hebr) at H1500, Holm-sig in 5/7 (31-60) and 4/7 (61-90) treatment worlds, ns in weak-context in both sets and ns in one of the two sets for agency / agency-briefs / big-pool (big-pool 61-90 p=.0504); at H500 2/7 and 0/7; it is behind in the first ~100 tasks (-.010..-.020 in 6 of 8 treatment/stress worlds, mostly ns) and not faster to a quality target. Tuned with a 16-point grid vs 3 for warm; the keyword pseudo-count (kw_w 8) is what keeps it near keyword early.
3. Controls: the retrieval layer is a bit-identical no-op when d = 0 (so "no regression" there is true by construction, not evidence of robustness) and hier ties best fair in ctl-coding-math/ctl-equal-agents (ns) and is ns-positive in ctl-nocontext-design (+.013/+.006). BUT `session(safe+retr)` (front = defaults::recommended_router) is a Holm LOSS vs the best fair baseline in ctl-coding-math (-.017/-.023 H1500, -.056/-.060 H100), ctl-nocontext-design (-.019, 31-60) and low-recurrence (-.018/-.016): guarded-safe is simply 0.01-0.02 worse than warm(hebr) in these graded, drift-free worlds (f7 saw the opposite in drift/stress worlds). Retrieval is not the cause (F2 shows 0.000).
4. Early cost: session(warm+retr) vs the early leader guarded-safe at H100: confusable -.058/-.045 (Holm LOSS both sets, also H250 -.021 on 61-90), big-pool -.024 (61-90 Holm LOSS); hier at H100 big-pool -.020 (61-90). Retrieval needs a first graded success per project before a note exists.
5. Wrong-brief harm: where recurrence is low or projects confusable, design (highest d) loses up to -.025 (confusable, session warm+retr, 31-60) and -.011/-.013 (low-recurrence); overall retrieval is ~0 in low-recurrence (+.001/.000 vs the same router) only because design (-.012) and marketing (+.014) net out (31-60 domain means, warm vs session(warm+retr)).
6. Retrieval knobs (store .55, sim .75) were tuned on seeds 1-8 with stress worlds in the objective; they differ from the library defaults (.6/.6). I did not run the library defaults themselves; the nearest grid point (store .55, sim .55) is 0.017 lower in the tuning objective (tuning objective, store .55: .693 at sim .55, .710 at .65, .711 at .75, .699 at .85; flat between sim .65 and .75; whether the stress worlds, which are in the objective, pull it toward the conservative end was not isolated).
7. Best fair baselines were chosen per cell on the held-out seeds (conservative) but baselines got small grids only; I did not tune hebbian-robust internals. Ties among duplicate baselines (tuned vs e1) are normal.
8. `hier+retr` runs in an in-bin replica of `ResearchSession::run_task` (exactness proven in G5-A for safe/warm), because the session cannot take a custom router; proposal in results/g5_proposed_patch.md. Critic is OFF (f7 verdict), so no critic claim is made here.
9. All worlds mine; quality is a clamped additive-noise model; n_tasks 1500, REVEAL .6, delay 15, one world family. No real traffic. Time-to-target means are censored at 1501 (the t90/t95 oracle targets are mostly unreachable by baselines, hence the t@95%bf metric).
