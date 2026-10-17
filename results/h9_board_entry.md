## H9 (R5-A): warm-start priors for the router from previous deployments

All numbers are SYNTHETIC-until-validated-on-real-traffic (author-built generators). No claim of time dilation, neuromorphic, 10x or "smarter model". The router is a feature->agent routing layer; "prior" = what a new deployment inherits from the LOGGED outcomes of earlier deployments. Code: `src/bin/h9_priors.rs`, `src/bin/h9_common/mod.rs` (src/*.rs untouched; port proposal in `results/h9_proposed_patch.md`). Opt-in only; defaults unchanged.

### Verdict (one paragraph)
KEEP, as an opt-in CONDITIONAL lever, not a default: `pop-guard+conf` (the recommended guarded-safe router with its (feature,agent) counts seeded from a pool of K>=2 previous deployments' logs, a trust weight per feature learned online, and an informed fallback). It wins early-horizon (H50..H500) with Holm significance in 2 independent worlds from 2 different generators (A1 sim-Fixed, B1 multi-domain) on all three seed blocks, costs a bounded amount when the prior is wrong (worst mean loss -0.031 at H50, <= -0.0074 from H250 on, <= -0.002 at H2000) and ties in controls. The gain is large only when the pool is the SAME hidden world with trap keywords (A1: +0.08..+0.09 at H50); in a family with 10% re-drawn specialists (B1) it is +0.013..+0.02 at H50-H250; at 40% re-drawn (B2) / big pool (B3) it is +0.002..+0.009 and mostly not Holm-significant. As expected from the science report it FADES: cumulative gains shrink to ~+0.01 at H2000 and the 1000-2000 window is slightly negative vs the best baseline in A1 (-0.013). No late-horizon gain is claimed.

### What was built and what died (candidates; all tuned on seeds 1-8 only, see below)
* `pop-guard+conf` (KEEP): `PopGuarded` = `prior::Guarded<HebbianRouter>` (exactly the recommended default) with counts initialised from the pool prior (pseudo-counts m*mu, m*(1-mu); m from between-deployment consistency, cap 16, scaled by coverage K'/K, cells seen in < 50% of deployments dropped), prior part scaled by a per-feature trust weight w_f = sigmoid(logit .8 + prequential log-score difference prior-vs-data, discounted .97, + .5 x global term), and an "informed fallback": when the keyword rule is proven poor/absent, follow the best-LCB candidate that carries >= 8 trust-weighted pool pseudo-trials if its LCB beats the keyword UCB (else the cold inner learner). Bit-identical to `recommended_router` with an empty prior (selftest).
* `pop-guard` (no trust): same gains, bigger wrong-prior cost (A5 H50 -0.035..-0.051, B6 H250 -0.017..-0.021). The trust weight is what buys the bound.
* `pop-ts+conf` (Thompson sampling with pool-initialised cells + trust): DEAD. +0.03..+0.12 over its cold counterpart pooled-ts at H50 in related worlds, but 0.10-0.36 BELOW the best fair baseline everywhere (a posterior-sampling learner cannot match the keyword-guard default early). Not a candidate for anything.
* `pop-map` (keyword map corrected from the pool; ablation): best of all in A1/B4 at H50 (.829 vs .740; B4 .786 vs .703) but UNSAFE: A5 -0.25, B6 -0.22, B8 -0.16 vs best baseline; the tuner's best config still has J = -0.117.
* `kw-cal` (guarded-safe with kw prior strength calibrated from the pool's keyword accuracy): ties everything (|diff| <= 0.005, 0 Holm wins/losses). Calibrating the keyword prior's strength from a pool buys nothing measurable.
* Hierarchical shrinkage ablation (`noshrink` = plain pooled counts capped at M): within +-0.001 of the shrinkage version in every world. The shrinkage estimator is NOT what makes it work here; coverage rule + trust + informed fallback are. Disclosed negative.

### Setup (worlds fixed before any policy ran; `--show` run first, results/h9_show.log)
Pool = K=6 previous deployments from the same generator family (different seeds, disjoint seed range 1,000,100+), each = the DEFAULT router (guarded-safe) running 1000 tasks; only logged (feature, agent, reward) survives (a biased logging policy: poor cells get few samples; handled by the coverage rule; `kw_acc` style statistics are selection-biased, disclosed). Eval deployment: 2000 tasks, H in {50,100,250,500,2000} cumulative mean EXPECTED success, windows, regret, tasks-to-target (first t with trailing-50 success >= q x trailing-50 oracle, q=.95/.90, censored at 2000).
Generators: A = `sim::World` (Fixed/Random/NoTraps) + f3 `BigWorld` (reused read-only); B = my own g5-style multi-domain world (5 domains x 4 sub-skills, three different agent-pool structures: 10-agent agency, 20-agent big pool, 10-agent latent-factor; decoy keywords; optional drift/new agents/renamed vocabulary/blind feedback).
Worlds (role): A1 sim-fixed (Treat, related pool = same hidden skills), A2 sim-random (Control, skills redrawn per seed = randomized world), A3 sim-notraps (Control), A4 big32 (Control), A5 sim-mismatch (Stress: pool Fixed, eval Random = WRONG prior), B1 md-related rho .1 (Treat), B2 md-partial rho .4 (Treat), B3 md-big20 (Treat), B4 md-shift (Treat: POOL DISTRIBUTION SHIFT: eval has drift, 2 agents unknown to the pool, decoy .5, other domain mix), B5 md-heldout-family (Stress: pool agency, eval latent-factor pool + half the vocabulary renamed), B6 md-wrong (Stress: every specialist moved = WRONG prior), B7 md-equal (Control: 8 identical agents), B8 md-noinfo (Control: skills random per deployment, pool inconsistent = no-prior-info), B9 md-uninformative-fb (Control: the reward the router observes is an agent-independent coin).
Tuning only on seeds 1-8 and only on A1,A2,A4,B1,B2,B6,B7; A3,A5,B3,B4,B5,B8,B9 are NEVER seen by the tuner (generalisation worlds; A5/B5/B4/B3 are the held-out family / shift / mismatch tests). Objective = regret-averse (mean margin + mean negative margin vs best fair baseline, H50..H500). 208 configs total (popts_conf 24, popguard 48, popguard_conf 96, kwcal 6, popmap 16, eps 9, pts 3, fixed 6). Frozen picks: pop-guard+conf M16 m1=1 cov.5 inf=8 w0=.8 cg=.5; pop-guard M16 m1=1 cov.5 inf=8; kw-cal m3 tau.7; pop-ts+conf M32 cov.5; pop-map M8 nc10 d.15 m3; eps-greedy(g1,e0), pooled-ts(g1).
Baselines (best picked post hoc per world x metric on the evaluated block's mean = conservative): keyword, eps-greedy(tuned), pooled-ts(tuned), hebbian-robust, warm(hebr,k1), warm(hebr,k10), guarded-safe (= recommended_router), kw-ts(m4) (keyword prior as TS pseudo-counts). Paired 95% t-CI over 30 seeds, two-sided; Holm per family (G gain: Treat x H x 4 cands; L loss: Stress+Control; F/FL windows; X regret/ttt; D vs matched cold counterpart; B9 own family; TOST +-0.01 for controls).

### Headline numbers: pop-guard+conf minus BEST FAIR BASELINE (mean [95% CI]; W/w = Holm/raw win; blocks 31-60 | 61-90 | 91-120)
```
world  H     31-60                      61-90                      91-120
A1     H50   +.0916 [+.082,+.101] W     +.0815 [+.074,+.090] W     +.0814 [+.072,+.091] W
A1     H100  +.0868 W                   +.0742 W                   +.0771 W
A1     H250  +.0650 W                   +.0559 W                   +.0567 W
A1     H500  +.0412 W                   +.0366 W                   +.0379 W
A1     H2000 +.0094 w                   +.0103 w                   +.0108 w
A1     win1000-2000 -.0130 l            -.0128 l                   -.0095 ns       (fades; slightly negative late)
A1     ttt95 +157 / +133 / +117 tasks sooner (W,W,W)   [means: H50 .825 vs guarded .740, keyword .739, warm(hebr,k1) .695]
B1     H50   +.0127 [+.007,+.018] W     +.0201 [+.011,+.030] W     +.0135 [+.004,+.023] w
B1     H100  +.0159 W                   +.0201 W                   +.0151 w
B1     H250  +.0168 W                   +.0208 W                   +.0147 W
B1     H500  +.0129 W                   +.0160 W                   +.0117 W
B1     H2000 +.0077 W                   +.0076 w                   +.0034 ns
B2     H50-250 +.006/+.005/+.004 w      +.004/+.007 W/+.007 w      +.002/.000/+.001 ns
B3     H100  +.0082 w                   +.0080 w                   +.0056 w            (never tuned on)
B4 (pool shift) H50 +.0209 w            +.0100 ns                  +.0058 ns ; H2000 +.0055 w | -.0063 ns | -.0113 l ; win1000-2000 -.002 | -.014 l | -.022 l
```
Early-gain criterion (Holm WIN at H50/H100/H250 in >1 independent world): A1 and B1 on all three blocks (B1 91-120: H250 W, H50/H100 raw only); B2 (61-90) and B4 (31-60) add single-block wins. Holm losses in Treat worlds: 0 on all blocks.

### Wrong prior: the cost and its bound (pop-guard+conf vs best fair baseline; A5 = pool Fixed -> eval Random; B6 = every specialist moved)
```
worst mean loss over {A5,B6} per horizon, three blocks:  H50 -.0310 | -.0205 | -.0165 ; H100 -.0193 | -.0141 | -.0088 ; H250 -.0074 | -.0066 | -.0065 ; H500 -.0046 | -.0051 | -.0051 ; H2000 -.0017 max
worst 95% CI lower bound: -.0456 (31-60, A5 H50)   B6 alone: -.0025..-.0055 at H50, -.0052..-.0066 at H250 (Holm L in 61-90 H100-H500), -.0014..-.0017 at H2000
bound: loss <= ~0.03 (CI -0.046) at H50, <= 0.02 at H100, <= 0.007 from H250, <= 0.002 at H2000; i.e. it recovers (no persistent cost) and is 2-3x smaller than without the trust weight (pop-guard: A5 -.035..-.051 at H50, -.024..-.031 at H250).
```
My informal bound before the confirmation blocks was "<= 0.02 at any H": NOT met at H50 in A5 (-.031/-.0205/-.0165, i.e. 1 of 3 blocks exceeds it; met from H100 on). Thresholds were not pre-registered in a file; I report the numbers rather than a pass/fail.

### Controls (pop-guard+conf vs best fair baseline; blocks pooled in words)
A2 randomized world: |diff| <= 0.003 everywhere (ties). A3 no-traps (keyword already near oracle): -.0015..-.0032 at H50-H500 (raw l at H100/H250; no Holm loss). A4 big32: |diff| <= 0.0008. B7 8 identical agents: exactly 0. B5 held-out family: -.0007..-.0036 (raw l). B8 no-prior-info: -.0068..-.0098 at H50, -.0058..-.0060 at H100 (Holm L at H100 in 91-120), <= -.005 at H2000. TOST +-0.01 vs best baseline, Holm-significant equivalence in 19/25, 20/25, 23/25 control cells (A2,A3,A4,B7,B8 x 5 H). So controls are harmless to within ~0.01 but NOT all proven equivalent (B8 early cells and A3 H100-250 are the exceptions).
B9 uninformative feedback: every learner collapses toward random after the keyword UCB falls (keyword, a static map, wins by construction: pop-guard+conf about -.08 at H250, -.17 at H2000 vs keyword) — that is the learner's problem, not the prior's: vs its cold counterpart guarded-safe pop-guard+conf is +.004..+.005 at H50-H250 (raw w at H100 in all 3 blocks) and within +-.011 late. Uninformative-feedback worlds need a feedback-quality gate, which is out of scope here.

### Other results
* K (number of previous deployments, frozen picks, 61-90, `--expk`): K=1: no effect (no consistency estimate, m1=1: A1 +.004); K=2: A1 +.075, B1 +.012; K=4: A1 +.077, B1 +.016; K=8: A1 +.083, B1 +.017 (H50 pop-guard+conf); wrong-prior A5 H50 -.0175 (K=2), -.0178 (K=4), -.0180 (K=8); B6 -.007/-.006/-.003. Saturates at K≈4.
* Cost (`--time`, noisy shared box, single thread, includes simulator calls): guarded-safe 1.7 us/task, pop-guard+conf 3.3 us/task (unoptimised, string-keyed maps, trust computed per feature x agent in `select`), warm(hebr) 3.1, pooled-ts 3.6. State per feature row 2x guarded (data + prior counts) + 16 B trust + a shared static prior table (features x agents x 32 B). Prior fit for K=6: <0.1 ms.
* Prior quality diagnostics (static first-250 policy "follow the prior's argmax", seed 31): A1 .831 vs keyword .748 vs oracle .836; B1 .758 vs .696 vs .867; B2 .599 vs .689 (worse than keyword); A2 .420 vs .696; A5 .403 vs .696; B6/B8 far below keyword. So a pool prior is only worth following where the pool and the deployment really share structure; the trust weight + coverage rule are what keep the router from following it when they do not.

### NEGATIVES / LIMITS (read with the claim)
1. Everything is simulation; "previous deployments" are other seeds of generators I wrote; relatedness (A1: same hidden skills; B1: 10% of domains re-drawn) is a parameter I chose. The size of the gain is a property of that parameter: at rho .4 it is within noise. No evidence yet on real traffic.
2. The headline +0.08..+0.09 (A1) is the "same product, new tenant" case with trap keywords; the keyword-guard default already captures most of the available gain in B-style worlds (decoy .3), so the remaining headroom is +0.01..+0.02 there.
3. Gains are early-horizon and fade (cumulative +0.01 at H2000; late window negative in A1 and in the shift world B4 where the pool also cannot know the 2 new agents).
4. Pool-distribution shift (B4): early gain not reproducible across blocks (+.021 w, +.010 ns, +.006 ns) and late window -0.002..-0.022 (Holm l in 61-90/91-120): the lever should not be relied on under drift + unknown agents.
5. Hierarchical shrinkage did not matter (ablation within +-0.001); `kw-cal` and `pop-ts` bring nothing; `pop-map` is unsafe. The surviving mechanism is "seed the guard's counts + trust-gate + informed fallback".
6. Logged pool data are selection-biased by the logging router (guarded-safe); the coverage rule is a heuristic fix, not a debiasing guarantee. A pool built from a different logging policy was not tested.
7. A wrong prior costs up to ~0.03 at H50 (CI -0.046) and ~0.006 around H250 in B6 (Holm-significant in 61-90); not free.
8. Uninformative-feedback deployments are harmful for ALL learned routers vs a static keyword map (B9); the prior neither causes nor fixes this.

### PROTOCOL DISCLOSURE (process honesty)
* Worlds A1-B8 were written and `--show`n before any policy ran. B9 (uninformative feedback) was ADDED after I had seen 31-60 results of the first design, to cover the "uninformative feedback" control; it is never used for tuning.
* Design iterations seen on 31-60: (1) first quick runs showed pop-ts catastrophically below baselines and pop-map unsafe (selection-biased pool cells), after which I (i) added the coverage rule `cov_min`, (ii) replaced the Thompson/map candidates by `PopGuarded` (seeded counts), (iii) added the informed fallback, (iv) added per-agent fallback `ma` for pop-ts (tuner chose ma=0). 31-60 is therefore NOT a clean held-out block for the final design; 61-90 and 91-120 are fresh, were first run with the final frozen tuning (deterministic on seeds 1-8) and were not looked at before. Conclusions rest on 61-90 and 91-120 (and agree with 31-60).
* tasks-to-target changed once (initial q=.85 saturated at the 50-task window floor; replaced by q=.95/.90 before the final runs); regret@250 dropped from the table.
* The wrong-prior bound was not pre-registered in a file (see above). Best baseline per (world, metric) is picked post hoc on the same block's mean (conservative, favours baselines). kw-ts and warm(hebr,k10) were added to the fair-baseline set (stronger than the spec'd list).
* Holm is per family and per block; the three blocks are not pooled. BigWorld is cut at 2000 tasks (pre-drift).

### VERIFY REQUEST — R5-A — H9 pool-seeded guarded router (`pop-guard+conf`)
Build/run (deterministic; ~1 min per block with F3_THREADS=2; a rerun of 61-90 reproduced the logged results exactly):
```
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r2 RAYON_NUM_THREADS=2 F3_THREADS=2
cargo build --offline --release --bin h9_priors
target\release\h9_priors --show                                   # worlds + prior diagnostics (results/h9_show.log)
target\release\h9_priors --held 31-60  --out results/h9_priors.json                      # log: results/h9_priors.log
target\release\h9_priors --held 61-90  --out results/h9_priors_confirm61-90.json         # frozen tuning, fresh seeds
target\release\h9_priors --held 91-120 --out results/h9_priors_confirm91-120.json        # second fresh block
target\release\h9_priors --held 61-90 --expk --out results/h9_priors_k_sensitivity61-90.json
target\release\h9_priors --time                                   # results/h9_cost.log
```
Self-check at start of every run (panics if violated): PopTs(empty prior) == pooled-ts bit-identical curves; PopGuarded(empty prior, trust on/off, informed on) == `defaults::recommended_router` bit-identical on 3 worlds; shrinkage/coverage/trust unit assertions.
Expected: tuning lines `[tune] popguard_conf ... J=+0.00681 pop-guard+conf[M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5]`; SUMMARY block per candidate (early_win_worlds, Holm loss cells, WRONG-prior worst, TOST counts); per-world DIFF rows; per-seed arrays for every policy x metric are in the JSON (`worlds[].policies[].per_seed`) so the paired CIs can be recomputed independently. Tables of the headline cells: `results/h9_tables.md`.
Please check specifically: (a) A1 and B1 H50/H100/H250 Holm WINs on 61-90 and 91-120; (b) the wrong-prior numbers for A5/B6 (worst -0.031 at H50); (c) that the tuner never touched A3,A5,B3,B4,B5,B8,B9 (`spec.tune=false`, h9_common `specs()`); (d) the pool is built from seeds 1,000,100+ (disjoint from eval seeds); (e) that the logged-data selection-bias caveat does not undermine the K=6 pool (try a different logging policy if you want a stricter check).
Files: src/bin/h9_priors.rs, src/bin/h9_common/mod.rs, results/h9_priors*.json|log, results/h9_priors_k_sensitivity61-90.*, results/h9_cost.log, results/h9_show.log, results/h9_tables.md, results/h9_proposed_patch.md, results/h9_board_entry.md. cargo test --lib not run (no library change).
