# G12 (W13) board entry: LEARNED-POLICY ROUTER (tiny feed-forward net) vs keyword / eps-greedy / pooled-thompson / warm(hebbian-robust) / guarded-safe

Files (mine; `src/*.rs` untouched, no commits, no new crates/downloads): `src/bin/g12_annrouter.rs`, `src/bin/g12_common/mod.rs` (worlds, the net, arm factory, driver, counting allocator, 6 unit tests),
`results/g12_annrouter.{json,log}` (held-out seeds 31-60), `results/g12_annrouter_confirm61-90.{json,log}` (fresh seeds 61-90, SAME frozen tuning), `results/g12_cost.{json,log}` (ns/decision, bytes of state), `results/g12_proposed_patch.md` (nothing to patch).

HONESTY FRAMING (encode in any downstream claim): everything here is a seeded simulation I wrote. An outer net cannot raise a frozen model's reasoning; this only measures ROUTING (which of a fixed set of agents gets a task).
Nonlinear worlds are nonlinear BY CONSTRUCTION (I built the XOR / RBF skill structure so additive routers are blind to it); they show what the net CAN do where such structure exists, not that real task streams have it.

## What was built
- `AnnRouter` (std-only, f32): input = 256 hashed token slots (multi-hot of the task's features) + optional dense continuous features + bias -> 1 tanh hidden layer (H tuned in {8,24,64}) -> one logit per agent. Online SGD on the SAME binary graded outcome every other arm sees (select -> realised Bernoulli outcome -> update).
  Mode `Reg` (used by every selected config): per-agent success-probability regression (BCE on the chosen agent's output only, bandit feedback), softmax(p/tau) exploration with tau halving gap tau0 .3 -> tau_min (decaying temperature), optional replay buffer (ring, cap 1000, k extra SGD steps/update).
  Mode `Pg`: REINFORCE with running-mean baseline (in the grid, never selected). `ann+kw` variant: adds a learned scalar skip connection from keyword-map votes to the logits (the analogue of how warm/guarded get the keyword prior). `ann-tok`: same net with the continuous features removed (equal-information ablation).
- Continuous features cannot ride in `Task::features` (strings; the baselines would treat each float as a new keyword), so they sit in a side table indexed by `Task::id`; the baselines see the SAME world through 8 equal-width bin tokens `x{j}:b{k}` (information asymmetry, controlled by `ann-tok`).
- Worlds (10): existing `sim::World` Fixed / Random / NoTraps (T = 2000 / 6000 / 20000, drift at T/2 as in sim.rs), f3 `Big` and `BigTail` (32 agents x 48 types; my copy parametrises T, bit-identical to f3_common at T=4000, unit-tested against `f3_common::Env`), plus 5 new synthetic worlds
  (hidden structure drawn per seed): `lin-cont` (6 continuous dims, skill .5+.42 tanh(w.x+b): additive/linear), `xor-cont` (agent 0 wins on sign(x0 x1)>0, agent 1 on <0, 4 flat agents, 4 noise dims), `xor-cont-drift` (roles swap at T/2), `xor-tok` (token-only XOR, equal information for everyone), `rich-cont` (10 dims, 3 informative, 8 agents, RBF bumps).
  Synthetic keyword maps = per token the agent with the best average true success over the tasks carrying it (an informed marginal prior: right when skill is additive, blind to interactions). World definitions were fixed (and `rich-cont` re-scaled once because its oracle was only .40) BEFORE any policy comparison was read.
- Arms: the six requested (keyword, eps-greedy tuned, pooled-thompson tuned, hebbian-robust, warm(hebr) k tuned, guarded-safe = `defaults::recommended_router`), `ann`, plus references: `pts-pairs` (pooled-thompson over single tokens AND all pairwise token conjunctions; a cheap NON-neural way to see 2-way interactions; NOT one of the requested baselines, so it is NOT in "best fair"), `ann-tok`, `ann+kw`, `ann` with the other regime's config.
- Protocol: all knobs tuned on seeds 1-8 ONLY, per regime (low: T=2000, Big 4000; mid 6000; high 20000 re-uses the mid config). Objective = mean over the 10 worlds of mean full-run success (equal world weights, ONE global config, like the baselines).
  Grids: eps-greedy gamma {1,.99,.97,.95} x eps {0,.01,.02,.05,.1}; pts gamma {1,.995,.98,.95}; warm k {.5,1,3,6}; net 171 configs (Reg: H {8,24,64} x lr {.05,.15,.4} x tau_min {.02,.05} x half {100,300,1500} x replay_k {0,3,8}; Pg: H x lr {.02,.05,.15}), tuned separately for `ann` and `ann+kw`.
  Selected (identical in the held and confirm runs, i.e. tuning is deterministic): low and mid/high `ann` = Reg H24 lr .05 tau .3->.02 half 100 replay_k 3; eps (.97,0) low / (.99,0) mid; pts gamma 1; warm k 6 (grid EDGE); `ann+kw` Reg+kw H8 lr .05 (low) / .15 (mid).
  Evaluation: held 31-60 (default) and 61-90 (confirm), 30 seeds, paired 95% t-CI of `ann` vs the BEST FAIR baseline (best mean among the six requested, picked per world x regime on the evaluated seeds: conservative against the net), for the full-stream mean and the last-25% mean. Holm within each regime family (20 tests per F1 table, 60 per metric-family overall).
  A claim is kept only if Holm-WINs come from >= 2 INDEPENDENT world families (sim, big, lin-cont, xor-cont(+drift), xor-tok, rich-cont). F1 = ann vs best fair; F2 = ann vs each of the 6; F3 = ann-tok vs best fair; F4 = ann vs pts-pairs; F5 = ann+kw vs best fair.

### VERIFY REQUEST — W13 — G12-A: unit tests (world fidelity + net sanity)
rerun: `export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r1 RAYON_NUM_THREADS=2; cd /c/Users/mariu/neural-mesh; cargo test --release --offline --bin g12_annrouter` (6 pass, <1 s after compile)
claim: `BigW` == `f3_common::Env::big / big_tail` bit-for-bit at T=4000 (tasks, probs, outcomes), `WorldSpec::Sim(Random).make(seed, 2000)` == `f3_common::Env::sim`, the net is deterministic, learns a single best agent (>80% of the last 500 picks), solves token XOR (net .712 vs additive pooled-thompson .543 late success, pairs-thompson .798), PG mode runs.

### VERIFY REQUEST — W13 — G12-B: main result (held 31-60 and confirm 61-90)
rerun (deterministic; tuning on 1-8 re-runs inside each call, ~20 min each at RAYON_NUM_THREADS=2; compare the numeric tables, `runtime_s` differs):
  `cargo run --release --offline --bin g12_annrouter -- --held 31-60 --out results/g12_annrouter.json`
  `cargo run --release --offline --bin g12_annrouter -- --held 61-90 --out results/g12_annrouter_confirm61-90.json`
(`--quick` = 3-seed smoke test, ~10 s.) Tune lines in both logs are identical (`[tune low]`, `[tune mid6000]`).

Mean expected success of the chosen agent over the whole stream (held 31-60; confirm 61-90 agrees within +-.01 everywhere, see files). `ann` diff = ann minus best fair baseline, paired 95% CI. Oracle = best agent per task.
```
regime low (T=2000, Big 4000)
world                 best-fair(mean)        ann    ann-tok  pts-pairs  oracle   ann-bestfair [95% CI]          Holm
sim-fixed             warm .7687            .7017   .7017    .7298      .8314    -0.0670 [-.0770,-.0570]        LOSS
sim-random            safe .7758            .7148   .7148    .7359      .8364    -0.0609 [-.0700,-.0519]        LOSS
sim-no-traps          safe .7940            .7212   .7212    .7321      .8354    -0.0727 [-.0795,-.0659]        LOSS
big-32agents          safe .7132            .5265   .5265    .5332      .8508    -0.1868 [-.2005,-.1731]        LOSS
big-32agents-longtail safe .6971            .5114   .5114    .5323      .8508    -0.1857 [-.2057,-.1657]        LOSS
lin-cont              safe .6887            .7185   .6575    .6722      .8136    +0.0298 [+.0234,+.0362]        WIN   (ann-tok loses: info asymmetry)
xor-cont              keyword .5478         .5490   .4917    .5219      .7768    +0.0012 [-.0103,+.0127]        ns    (late-25%: +.0897 WIN)
xor-cont-drift        keyword .5434         .4894   .4889    .5048      .7768    -0.0540 [-.0590,-.0489]        LOSS
xor-tok               keyword .5549         .6771   .6771    .7188      .8800    +0.1222 [+.1023,+.1421]        WIN   (vs pts-pairs: -.046, ns)
rich-cont             safe .5421            .5688   .5386    .5407      .7241    +0.0264 [+.0196,+.0332]        WIN
regime mid (T=6000)
sim-fixed             safe .7896            .7531   .7531    .7652      .8318    -0.0365 [-.0437,-.0293]        LOSS
sim-random            safe .7938            .7507   .7507    .7635      .8353    -0.0431 [-.0530,-.0333]        LOSS
sim-no-traps          safe .8034            .7495   .7495    .7601      .8334    -0.0539 [-.0655,-.0424]        LOSS
big-32agents          safe .7198            .5533   .5533    .5689      .8509    -0.1666 [-.1779,-.1553]        LOSS
big-32agents-longtail safe .7079            .5362   .5362    .5729      .8509    -0.1717 [-.1891,-.1542]        LOSS
lin-cont              eps .7073             .7492   .7006    .7153      .8142    +0.0419 [+.0334,+.0504]        WIN
xor-cont              keyword .5390         .6555   .5954    .5639      .7773    +0.1166 [+.1021,+.1310]        WIN
xor-cont-drift        keyword .5342         .5868   .5318    .5181      .7773    +0.0526 [+.0420,+.0632]        WIN
xor-tok               eps .5782             .7728   .7728    .7743      .8800    +0.1946 [+.1702,+.2191]        WIN   (vs pts-pairs: -.001, ns)
rich-cont             eps .5709             .6087   .5833    .5743      .7238    +0.0378 [+.0294,+.0462]        WIN
regime high (T=20000)
sim-fixed             safe .8011            .7731   .7731    .7809      .8319    -0.0280 [-.0334,-.0225]        LOSS
sim-random            safe .8035            .7820   .7820    .7796      .8366    -0.0215 [-.0327,-.0104]        LOSS
sim-no-traps          safe .8188            .7950   .7950    .7840      .8374    -0.0239 [-.0308,-.0169]        LOSS
big-32agents          safe .7517            .6285   .6285    .6745      .8509    -0.1231 [-.1361,-.1102]        LOSS
big-32agents-longtail safe .7462            .6089   .6089    .6755      .8509    -0.1373 [-.1518,-.1227]        LOSS
lin-cont              pts .7366             .7635   .7216    .7467      .8141    +0.0270 [+.0247,+.0292]        WIN
xor-cont              eps .5403             .7174   .6973    .6299      .7773    +0.1771 [+.1650,+.1893]        WIN
xor-cont-drift        keyword .5305         .6851   .6575    .5334      .7773    +0.1546 [+.1432,+.1659]        WIN
xor-tok               eps .5953             .8315   .8315    .8014      .8800    +0.2362 [+.2164,+.2561]        WIN   (vs pts-pairs: +.030, ns)
rich-cont             pts .5954             .6351   .6167    .6127      .7232    +0.0396 [+.0357,+.0436]        WIN
```
Holm WIN/LOSS counts (F1 full-stream metric, per regime, 10 worlds each): held low 3W/6L, mid 5W/5L, high 5W/5L; confirm low 3W/6L, mid 5W/5L, high 5W/5L (late-25% metric: held 4W/6L, 5W/5L, 5W/3L; confirm 4W/5L, 5W/3L, 5W/4L).
Independent-family rule: WIN families at every regime >= 3 (low: lin-cont, rich-cont, xor-tok; mid/high: + xor-cont) in BOTH splits, so the claim "the net beats the best requested baseline in worlds with continuous features and/or interactions" is KEPT. LOSS families: sim and big at every regime (and xor-cont-drift at low), in both splits.
Reference rows: ann vs pts-pairs (F4): net wins lin-cont, xor-cont(+drift), rich-cont at mid/high in both splits; it does NOT beat the conjunction-token baseline in token XOR (xor-tok: -.042/-.046 low, ~0 mid, +.03/+.04 high, all ns) and LOSES to it in sim at 2000 tasks (sim at 6000-20000: mostly ns, 1 LOSS + 1 WIN in the held split) and in Big at every volume (-.007..-.067, LOSS at 6000/20000 in both splits). F5 (`ann+kw` vs best fair): same pattern as `ann`; it closes roughly half (sim) to two-thirds (Big) of the gap (e.g. big low: .6465 vs ann .5265, safe .7132) but still LOSES every sim/big cell in both splits (-.017..-.077).

### VERIFY REQUEST — W13 — G12-C: data-volume crossover (tasks-to-parity curves; high regime, windows of 500 tasks, ann vs the late-quarter-best fair baseline, 30 seeds)
(printed as `curve vs ...` lines in the logs, window arrays in `curves` of the json). Parity = ann window mean >= ref - .005 and stays so for ALL later windows; "CI>0" = per-window paired 95% CI lower bound > 0 for all later windows. Confirm 61-90 in brackets.
```
world                  tasks-to-parity   sustained CI>0   full-stream gap vs best fair at T=2000 / 6000 / 20000
sim-fixed/random/no-traps  never (<=20000)  never          -.067,-.061,-.073 / -.037,-.043,-.054 / -.028,-.022,-.024  (gap closes ~3x, no crossing)
big-32 / longtail          never            never          -.187,-.186 / -.167,-.172 / -.123,-.137
lin-cont                   500 [500]        500 [500]      +.030 / +.042 / +.027  (ann-tok, equal info: LOSS at 2000, ~tie by 20000; late-25% margin over pts shrinks to +.011..+.013 at 20000)
xor-cont                   1500 [1500]      1500 [1500]    +.001 (late-25% +.090) / +.117 / +.177
xor-cont-drift             11500 [11500]    11500 [11500]  -.054 / +.053 / +.155  (ahead early, DIPS right after the T/2 swap, re-learns by ~11.5k)
xor-tok                    500 [500]        1000 [1000]    +.122 / +.195 / +.236  (vs conjunction baseline pts-pairs: ns at every volume)
rich-cont                  500 [500]        500 [500]      +.026 / +.038 / +.040  (margin GROWS with data)
```
Reading: (i) hypothesis "ties or loses at low data" is CONFIRMED only for the existing keyword-token worlds (sim/Big, which are additive + drift/failure and where the keyword prior matters) and for xor-cont (tie at 2000, crossover ~1000-1500 tasks on the late metric); it is REFUTED for worlds with rich/continuous or interaction structure, where the net is ahead within the first 500 tasks. (ii) "edges ahead only with lots of data" is NOT supported: in rich-cont the margin does grow with data (+.026 -> +.040), but nothing needs 6000+ tasks to flip sign. (iii) NO volume up to 20000 produced a crossover in the existing worlds: gap vs the best fair baseline narrows (sim -.07 -> -.025, Big -.19 -> -.13) but stays Holm-significant in all 3 sim and both Big worlds at every volume, both splits. Do not extrapolate beyond 20000.

### VERIFY REQUEST — W13 — G12-D: cost (bytes of state, ns/decision)
rerun: `cargo run --release --offline --bin g12_annrouter -- --timing --out results/g12_cost.json` (single thread, min of 3 reps, shared box so +-30% noise; ~1 min). ns/decision = select+update net of an empty-router loop (~10-15 ns).
```
                          ns/decision  sim-fixed(6 ag) | big(32 ag) | xor-cont   | rich-cont      live heap after run T=2000 -> T=20000 (sim-fixed / big / rich-cont)
keyword                          172-253 | 189-258 | 265-330 | 404-639            2.5 kB->2.7 kB / 9.5->9.8 kB / 4.7->5.0 kB
eps-greedy                     2124-2351 | 13.3k    | 4.4k    | 9.0-9.2k          5.7->6.0 kB / 138 kB->138 kB / 16.8->17.1 kB
pooled-thompson                2696-3079 | 15.4-18.6k | 5.1k  | 10.0-10.3k        5.7->5.9 kB / 138->138 kB / 16.8->17.1 kB
hebbian-robust                 1875-2150 | 7.1-7.5k | 3.7-4.0k | 7.7-7.9k          6.1->6.4 kB / 92->92 kB / 16.4->16.7 kB
warm(hebr)  (grows with T)     3264-3545 | 8.1-8.4k | 6.1-6.5k | 9.8-10.1k         87 kB->310 kB / 272 kB->1.18 MB / 318 kB->2.33 MB
guarded-safe                   1994-2198 | 5.4-5.7k | 4.9-5.0k | 9.0-9.2k          15 kB->15 kB / 246->246 kB / 38->38 kB
pts-pairs [ref]                7741-8396 | 38.8-41.4k | 27.5-28.3k | 96-99k       92->98 kB / 1.77->7.89 MB / 631 kB->631 kB
ann H24 replay_k3 (selected)   2206-2355 | 4216-4320 | 3349-3533 | 4191-4298     120 kB->120 kB (flat; 25-28 kB params + replay ring up to 1000 x 92 B, Vec capacity 1024)
ann H8 no-replay               588-629   | 1221-1244 | 827-845 | 1138-1159       8.8-9.4 kB params only, flat
ann-pg H24                     983-1060  | 2331-2484 | 1369-1400 | 1798-2009      25.6-28.5 kB, flat
```
State size analytics: H24 net 6.3k-7.0k f32 params = 25-28 kB (H8: 2.1-2.3k params, 8.4-9.4 kB); replay ring adds up to 92 B x 1000 (the `replay_entry_bytes` field in the json under `nets` says 80: that analytic field undercounts; the live-heap column is the measured one). State is CONSTANT in T (unlike warm(hebr), pts-pairs). Decision cost: on the 6-agent world the selected replay config (~2.2 us) is on par with eps/guarded; on the 32-agent world it (4.2 us) is below eps/pts/hebbian/warm/guarded (5.4-18.6 us); the selected replay_k=3 config costs ~4x an H8 no-replay net.

## Negatives / caveats (read before using any of this)
1. Existing keyword-token worlds (the ones this crate's routers were built for): the net LOSES to the best fair baseline in all 5 worlds x 3 volumes x 2 seeds-splits (Holm). Even `ann+kw` with a keyword-prior skip loses (-.017..-.077). The recommended default stays guarded(hebbian-robust); this experiment gives no reason to replace it.
2. The wins live in worlds I built to contain what a net exploits (continuous features, XOR / RBF structure). They are not evidence about real task streams. `lin-cont` is an information-asymmetry win: with the continuous features removed (`ann-tok`, equal information) the net LOSES at low volume (.6575 vs best fair .6887 at 2000) and only ties by 20000. So "linear world: the net should not help" holds under equal information.
3. A cheap non-neural alternative gets most of the token-XOR gain: `pts-pairs` is as good as the net at >= 6000 tasks and better at 2000 (-.046, ns after Holm). The net's decisive edge over it is on continuous-feature worlds (xor-cont +.08..+.09, rich +.02, drift +.15) where the pair-of-bins features are coarse.
4. `xor-cont-drift`: after the T/2 role swap the net dips below the best fair baseline until ~11.5k tasks; the replay buffer (cap 1000, k=3) replays pre-drift samples. The selected config has no forgetting mechanism beyond constant-lr SGD.
5. Tuning limits: ONE global config per regime tuned on the equal-weight mean over 10 worlds (nonlinear worlds contribute large gaps, the net's choice may be a compromise); grid edges hit for the net (`half`=100 is the grid minimum) and for warm (k=6 is the maximum) and eps (eps=0 is the minimum), so those arms could improve with wider grids. Hashing of 256 token slots collides in `BigTail` (3000-token tail vocabulary) and capacity H<=64 is small for 32 agents x 48 types; a bigger net might narrow the Big gap (not tested).
6. Rewards are the binary realised outcome, no delay, no failures beyond what sim.rs has (the sim failure window is a nonstationarity the net handles worse than the discounted counters). Single-agent solo routing only (team tasks not used). Timing is single-thread on a shared box.
7. "Best fair baseline" is chosen on the same evaluated seeds (selection favours the baseline); `pts-pairs`, `ann-tok`, `ann+kw` are reported as references, not as part of the claim.
8. The data-volume crossover is a property of these worlds; sim worlds redraw drift at T/2 per T, so the T=2000/6000/20000 numbers are separate world instances, not prefixes of one stream. The tasks-to-parity curves come from the T=20000 run only.
