# H12 (R5-D) board entry: opt-in ONLINE ADAPTATION layers (residual net / config bandit / meta-selector), honest neural re-test
Label on every number: SIMULATION, synthetic-until-validated-on-real-traffic. Routing only (which fixed agent gets a task); no model-IQ or weight-learning claim. No src/*.rs edits, no commits, no new crates.
Files (mine): src/bin/h12_online.rs, src/bin/h12_common/{mod.rs (copy of g12 worlds+net, Env/WorldSpec extended), drift.rs (new worlds), online.rs (the layers)},
results/h12_online.{json,log} (held 31-60), results/h12_online_confirm61-90.{json,log} (fresh 61-90, SAME frozen tuning; tune lines identical), results/h12_cost.{json,log}, results/h12_proposed_patch.md, results/lessons_h12.md.

## Verdict (blunt)
- Residual neural layer on the incumbent (resid-warm, resid-safe): REAL gain ONLY in worlds that contain interaction structure (vs best fair: xor-cont +0.09..+0.10, xor-cont-drift +0.04..+0.08, xor-drift3 +0.00..+0.03, xor-tok +0.21; held AND confirm, Holm-WIN). It is a tie or Holm-LOSS in every keyword-prior world (sim x3, big, kw-drift4) and in both controls, and it does NOT win lin-cont or rich-cont (loses to tuned eps-greedy in rich-cont, -0.03). By the pre-declared rule (Holm-WIN in >=2 independent families AND TOST +-0.01 harmless in all 7 keyword/control worlds) it is NOT KEPT: it passes the first half (families xor-cont and xor-tok, both splits) and fails the second. Bounded loss vs its own incumbent in keyword/control worlds is at most about -0.011 mean (resid-safe: ctl-static -0.0106 held / -0.0108 confirm; resid-warm worst -0.012 held kw-drift4, -0.006 confirm); worst 95% CI-lo -0.014..-0.018; TOST fails in 2-4 of 7 worlds. If the director accepts "documented bounded loss" in place of TOST, resid-safe is a defensible OPT-IN for streams KNOWN to have feature interactions; I recommend nothing beyond that, and never as default.
- Config bandit (online hyper-parameter adaptation over 6 warm(hebr) configs): REJECTED. 0 Holm wins vs best fair anywhere; Holm-LOSS in xor-cont/xor-tok/rich/ctl/sim cells. vs its own incumbent warm(hebr): small gains (+0.005..+0.009 sim-fixed, +0.021..+0.026 lin-cont, +0.010..+0.014 rich, +0.006..+0.009 ctl-static) but -0.006..-0.012 in kw-drift4 and ties elsewhere; no significant recovery gain after drift; 6x state and ~4-7x ns.
- Meta-selector (bandit over {keyword, warm(hebr), guarded-safe, resid-warm}): NOT KEPT. Inherits the residual's wins in xor-cont/xor-tok (+0.10, +0.21 vs best fair) but pays an exploration tax: -0.004..-0.020 vs guarded-safe in sim/big/ctl-static (TOST fails 7/7 held, 6/7 confirm), worst CI-lo -0.032 held / -0.024 confirm. 4x state.
- "Adaptation over time" (recovery after drift) is NOT delivered by the neural layer: it raises the level but deepens the dip. xor-drift3 / xor-cont-drift: dip 0.07-0.18 (resid) vs 0.02-0.04 (incumbents), recovery 800-1400 tasks vs 300-450; its post-drift regret is still lower (-0.01..-0.055) only because its pre-drift level is much higher. In keyword drift worlds (sim, big, kw-drift4) recovery is the same as or slightly worse than warm(hebr). The causal gate is what keeps it safe: the no-gate ablation is a Holm-LOSS vs the gated layer in sim-no-traps, big, kw-drift4 (-0.015, -0.047, -0.055) and its xor-drift dip is 0.36.

## Setup (all fixed before any policy ran; T=6000 in every world)
Arms: fair baselines keyword, eps-greedy (tuned), pooled-thompson (tuned), hebbian-robust, warm(hebr) (k tuned), guarded-safe (= recommended_router). Candidates: resid-warm (layer over warm(hebr)), resid-safe (layer over guarded-safe), cfg-bandit, meta, ablation resid-warm-nogate.
Residual layer (online.rs): the incumbent decides first with the SHARED rng, so with the gate closed and e0=0 the layer is bit-for-bit the incumbent (unit test). Net = tanh hidden layer (H 8..24) on 512 hashed token slots + raw continuous features, output layer ZERO-initialised on top of a per-agent global base rate, trained by SGD + replay(3) on the realised Bernoulli outcome of the executed agent (same graded outcome every arm sees). Causal gate: no override before n_min observations; per agent, override allowed only if EMA(prequential logloss of a discounted token-additive predictor minus logloss of the net) > 0.005 over >=30 observations of that agent (held-in rolling validation, predict-before-update); override only if the net's predicted p for the candidate exceeds the incumbent pick's by `margin`. Optional exploration e0*0.5^(t/1000), uniform (counted as real cost; tuned in {0,.05,.15}). NOTE the net sees continuous features, the baselines only an 8-bin token cut (same information asymmetry as W13); xor-tok is the equal-information world.
Config bandit: 6 fixed warm(hebr) configs (base, eta .05, decay .05, eps .05, temp .02, slow), discounted-Thompson, ALL members learn from ALL outcomes, optional agreement-credit.
Meta: same bandit over {keyword, warm(hebr), guarded-safe, resid-warm}, optional prior favouring guarded-safe.
Worlds (13; g12 worlds copied + 4 new, new ones written in drift.rs before running): sim-fixed/random/no-traps, big-32agents, lin-cont, xor-cont, xor-cont-drift, xor-tok, rich-cont (copies of g12; a unit test checks equality with g12_common), NEW kw-drift4 (keyword world, 3 drift events, 40% of types re-draw their best agent, keyword map exact only in epoch 0), NEW xor-drift3 (xor-cont stream, role swap at every quarter), NEW ctl-noinfo (random tokens, static task-independent skills, garbage keyword map: negative control), NEW ctl-static (no drift, exact keyword map: keyword optimal). Independence families: sim, big, lin-cont, xor-cont (xor-cont, xor-cont-drift, xor-drift3 = ONE family), xor-tok, rich-cont, kw-drift, ctl-*.
Protocol: tune seeds 1-8 only (one global config, objective = mean over the 13 worlds incl. controls; grids: eps 20, pts 4, warm k 4, residual 48 per incumbent, cfg-bandit 6, meta 12), held-out 31-60, confirm 61-90 frozen (tune lines identical in both logs: resid-warm H8 lr.05 nmin100 margin.08 e0=.05; resid-safe H8 lr.05 nmin300 margin.08 e0=0; warm k=6 (grid edge); eps (.99,0); cfg-bandit gamma 1, no agree; meta gamma .998, no agree, prior 0). Paired 95% t-CI vs the best fair baseline per world (picked on the evaluated seeds, conservative), full-stream and last-25% means, Holm over 104 tests (4 candidates x 13 worlds x 2 metrics); Holm over 52 for candidate-vs-own-incumbent; TOST margin +-0.01 pre-declared, implemented conservatively as "95% CI inside +-0.01".

### VERIFY REQUEST — H12 — unit tests
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r5 RAYON_NUM_THREADS=2; cd /c/Users/mariu/neural-mesh; cargo test --release --offline --bin h12_online
5 pass: copied worlds == g12 worlds, new worlds sane (static: keyword ~optimal), closed-gate layer == incumbent bit-for-bit, residual deterministic and explores, portfolios run.

### VERIFY REQUEST — H12 — main result (held 31-60, confirm 61-90; ~12 min each at 2 threads, deterministic; tuning reruns inside each call)
cargo run --release --offline --bin h12_online -- --held 31-60 --out results/h12_online.json
cargo run --release --offline --bin h12_online -- --held 61-90 --out results/h12_online_confirm61-90.json   (--quick = 3-seed smoke, ~2 min)
Mean expected success over the stream (HELD 31-60; confirm agrees within ~+-.01 in every cell, see log). Oracle and best fair last.
```
world          keyw   eps    pts    hebr   warm   safe  | r-warm r-safe cfgbd  meta  nogate| oracle  best-fair
sim-fixed      .7018 .7674 .7584 .7665 .7790 .7896 | .7787 .7844 .7881 .7807 .7787 | .8318 safe
sim-random     .6932 .7663 .7565 .7759 .7853 .7938 | .7780 .7881 .7869 .7822 .7676 | .8353 safe
sim-no-traps   .7901 .7599 .7542 .7663 .7864 .8034 | .7840 .7983 .7891 .7940 .7691 | .8334 safe
big-32agents   .6265 .5783 .5596 .6002 .6974 .7198 | .6923 .7160 .7038 .6995 .6454 | .8509 safe
lin-cont       .6693 .7073 .7034 .6523 .6720 .6882 | .6819 .7006 .6974 .6886 .7229 | .8142 eps
xor-cont       .5390 .5300 .5136 .5171 .5209 .5240 | .6334 .6393 .5182 .6369 .6766 | .7773 keyword
xor-cont-drift .5342 .5199 .5142 .5181 .5218 .5233 | .5767 .6124 .5196 .6015 .5881 | .7773 keyword
xor-tok        .5406 .5782 .5258 .5290 .5321 .5419 | .7905 .7891 .5355 .7914 .7854 | .8800 eps
rich-cont      .5305 .5709 .5622 .5322 .5303 .5400 | .5330 .5417 .5442 .5374 .5799 | .7238 eps
kw-drift4      .6457 .6902 .6895 .7580 .7756 .7543 | .7636 .7460 .7634 .7531 .7091 | .8493 warm
xor-drift3     .5303 .5203 .5112 .5184 .5199 .5211 | .5330 .5562 .5203 .5399 .5171 | .7775 keyword
ctl-noinfo     .4816 .7437 .7345 .7416 .7369 .6767 | .7353 .6762 .7339 .7248 .7417 | .7500 eps
ctl-static     .8501 .7946 .7872 .7897 .8192 .8491 | .8179 .8385 .8251 .8383 .7994 | .8501 keyword
```
Holm-WIN vs best fair (full metric), held AND confirm: resid-warm / resid-safe / meta: xor-cont, xor-cont-drift, xor-tok (+0.21); xor-drift3 for resid-safe and meta only; cfg-bandit none. Holm-LOSS vs best fair: sim x3 (-.005..-.02), ctl-static (-.012..-.032), ctl-noinfo (resid-safe -.068 because guarded-safe's keyword prior is garbage there; resid-warm -.008), rich-cont (-.027..-.038), kw-drift4 (-.012..-.030), lin-cont (resid-warm, meta -.019..-.025). Both controls are LOSSES for every candidate against the best fair baseline; vs the layer's own incumbent resid-warm is EQUIV in ctl-noinfo / ctl-static (TOST ok, both splits), resid-safe loses -0.0106/-0.0108 in ctl-static (keyword rule optimal there; the open net/exploration adds noise).
Gate ablation (nogate vs gated resid-warm, held): big -0.047, kw-drift4 -0.055, sim-no-traps -0.015 (LOSS); lin-cont +0.041, xor-cont +0.043, rich-cont +0.047 (WIN): the gate trades the continuous-world gain for safety, as designed.

### VERIFY REQUEST — H12 — adaptivity and tasks-to-parity (logs: `ADAPTIVITY`, `TIME-TO-PARITY`; curves with per-window CIs in the JSON)
Recovery after drift (regret to oracle in 250-task bins; post window 1500 tasks; dip = peak regret minus pre-drift regret; recovery = tasks until regret <= pre+.03, censored at 1500). HELD; confirm similar (e.g. xor-cont-drift confirm resid-safe dip .171 / rec 1300 vs guarded-safe .029 / 475):
```
world (events)              warm(hebr) dip/rec   guarded-safe dip/rec   resid-safe dip/rec   cfg-bandit dip/rec   post-drift regret resid-safe vs best fair [95% CI]
sim-fixed (3000)            .059 / 883           .079 / 1300            .074 / 1167          .056 / 783           +.008 [-.002,+.017] tie
big-32 (3000)               .074 / 925           .091 / 1158            .088 / 1283          .084 / 1050          +.001 [-.010,+.013] tie
kw-drift4 (1500,3000,4500)  .115 / 892           .139 / 1119            .134 / 1094          .131 / 961           +.043 [+.029,+.057] worse (best fair = warm(hebr))
xor-cont-drift (3000)       .023 / 325           .028 / 450             .184 / 1400          .021 / 317           -.043 [-.052,-.034] lower regret, much deeper dip
xor-drift3 (1500,3000,4500) .022 / 325           .024 / 408             .073 / 828           .024 / 489           -.040 [-.046,-.033] idem
```
cfg-bandit recovers marginally faster in sim-fixed (-.008 post-drift regret vs best fair, CI touches 0) and nowhere else significantly. The residual layer helps the post-drift LEVEL in interaction worlds but is slower to adapt (constant-lr SGD plus replay of pre-drift samples; gate must close/reopen): it is not a drift-adaptation mechanism.
Tasks-to-parity vs the late-best fair baseline (T=20000, windows of 500, parity = within .005 sustained to the end; held [confirm]): resid-warm/resid-safe/meta: xor-tok 500 [500]; xor-cont 500 [500..1500], sustained CI>0 from 1500-2000 [1500]; xor-drift3 500 [500] (sustained CI>0 only from 16000 for resid-safe); NEVER within 20000 in lin-cont, rich-cont, kw-drift4, sim-fixed (resid-*, meta); ctl-noinfo parity at 4000 (resid-warm, held) but none on confirm. cfg-bandit: parity at 500 [11000] in kw-drift4 (sustained CI>0 only from 17000-18500) and sim-fixed, none in xor/rich/lin worlds.

### VERIFY REQUEST — H12 — cost (rerun: cargo run --release --offline --bin h12_online -- --timing --out results/h12_cost.json; single thread, shared noisy box +-30-100%: trust ratios and bytes)
Live bytes after T=6000 / ns per decision net of loop (default layer config H24, replay 3):
```
                         sim-fixed (6 ag)        big (32 ag)             xor-cont (6 ag, 6 dense)  kw-drift4 (8 ag)
warm(hebr)               165 kB / 1.8 us         424 kB / 4.7 us         696 kB / 3.5 us           286 kB / 2.2 us
guarded-safe              15 kB / 1.2 us         246 kB / 3.3 us          20 kB / 2.7 us            34 kB / 1.6 us
resid-warm               322 kB / 3.0 us         691 kB / 6.2 us         891 kB / 5.4 us           452 kB / 3.7 us
resid-safe               172 kB / 2.3 us         513 kB / 4.6 us         178 kB / 4.9 us           199 kB / 4.5 us
cfg-bandit (6 members)   995 kB / 6.5 us         2.5 MB / 11.6 us       4.2 MB / 20 us            1.7 MB / 7.8 us
meta (4 members)         515 kB / 4.4 us         1.5 MB / 9.2 us        1.7 MB / 8.5 us           786 kB / 6.2 us
```
Layer cost over its incumbent: about +150-270 kB of state (flat in T except the 1000-entry replay ring) and +1.1-1.4 us/decision on 6 agents; portfolios multiply state and ns by the number of members (agree credit up to ~2.4x more ns).

## Negatives / caveats
1. Every win is in a world built to have interaction structure (by construction). xor-cont has an information asymmetry (net gets continuous x, baselines an 8-bin cut) as in W13. xor-tok is equal-information, BUT g12 showed a cheap non-neural pairwise-conjunction pooled-thompson reaches most of that gain (not re-run here, not part of the fair set).
2. No win in lin-cont (additive), rich-cont (smooth; tuned eps-greedy better), or any keyword world: consistent with W13. The "adaptation over time" framing is NOT supported: recovery after drift is worse for the net layer, and bandit-over-configs does not beat the best fair baseline anywhere.
3. Best-fair is chosen on the evaluated seeds (conservative against the candidates). One global config per candidate tuned on the equal-weight mean over 13 worlds incl. controls. Grid edges hit: warm k=6 (max), eps=0 (min), resid H8 (min), lr .05 (min), margin .08 (max); wider grids not explored. resid-safe tuned to e0=0, so it only learns from the incumbent's own exploration.
4. The gate's "beats the incumbent" is a predictive-loss comparison against a token-additive discounted predictor (the incumbent's analogue), not a counterfactual policy value (not identifiable from bandit feedback without propensities). Discount .98 per slot touch, beta .02, delta .005 were fixed a priori and not tuned. Hash collisions: 512 slots.
5. The residual wraps the incumbent behaviourally (override of its pick), not by adding to its internal scores: the Router trait exposes no per-agent scores for Warm/Guarded (see results/h12_proposed_patch.md). The incumbent still learns from the executed (possibly overridden) agent.
6. Rewards: immediate binary outcome, single-agent routing, no failures/delays beyond sim.rs. Timing single-thread on a shared box.
7. Disclosure: one 3-seed smoke run with a reduced grid (not reported) tested the pipeline; no world or arm was changed afterwards. No post-hoc changes.
