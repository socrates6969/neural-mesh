## G8 / W9 — spiking-network mechanisms (LIF/Izhikevich WTA, eligibility traces over delayed reward, population coding)

Files: src/bin/g8_snn.rs, src/bin/g8_common/{mod,snn,worlds,pol}.rs (no src/*.rs edits, no new crates, nothing committed).
Results: results/g8s_snn.json (held-out 31-60), results/g8s_snn_confirm.json (fresh 61-90, frozen configs), results/g8s_snn.log.
Superseded earlier runs kept for transparency: results/g8s_snn_v1_* (baselines had a much smaller tuning budget than the candidates; see "history"), results/g8s_snn_v2_* (matched-effort baselines, before the rate-ctl ablation rows were added; all shared rows are byte-identical to the final run).

### Verdict (honest)
- Spiking selectors (LIF-WTA, Izhikevich-WTA, with lateral inhibition, population-coded inputs, eligibility-trace three-factor learning): REJECTED. 0 Holm wins vs the best fair baseline in any counted world on either seed set; Holm-significant LOSSES in B, C and D (both sets), and an Izhikevich loss in A-late. Not recommended for merge into any default or opt-in path.
- The NON-spiking control (same encoder, same trace, same learner, softmax instead of spikes; label "rate-ctl") DID clear the statistical bar (Holm win vs best fair baseline in A and D, on held-out AND confirm), but the margin is ~+0.008 absolute (~1%) over a matched-effort baseline, it is NOT specific to delay or to population coding (it also beats the baselines by +0.013 in the static-immediate control world E, and loses in C where population coding should shine), and it was tuned per world. I would treat it as a lead for a separate evaluation, not as a result. It is not what the task asked to be kept (spiking), so the SNN verdict stays REJECTED.
- Does the spiking part matter? Yes, negatively: snn-lif minus rate-ctl is -0.006 (A), -0.025 (B), -0.008 (C), -0.025 (D), -0.009 (E) abs, 95% CI excluding 0 in all (held-out; confirm same signs). The tuned WTA selectors run saturated (fraction of decisions with >=1 spike = 1.00, 12-77 spikes per decision for LIF, ~1-8 for Izhikevich): they act as a noisy argmax, which is a worse exploration policy than a tuned softmax, and they cost more (ns/decision below).
- Eligibility traces over delayed reward: no demonstrated benefit. In B (anonymous geometric-delay reward, mean 6 steps) trace vs lambda=0 is not significant for snn-lif (-0.010 [-0.035,+0.016]) and for rate-ctl the lambda=0 ablation is slightly BETTER (+0.024 [+0.002,+0.046] held; +0.009 [-0.013,+0.032] confirm). Trace beats fixed-lag attribution (+0.077 [+0.056,+0.098], snn-lif) but every learner is far below the keyword rule in B (-0.10 to -0.16): credit dilution at 1/6 and 2500 tasks is too much for learning to overcome the keyword prior.
- Population coding of continuous features: no demonstrated benefit for the spiking selector (snn-lif -pop-code vs snn-lif: n.s. in D, +0.012 in C favouring pop-code but still -0.018 vs best baseline). For rate-ctl removing the pop code costs -0.006 (D) and -0.013 (C) abs, so the code is not useless, but in C (the smooth-bump world built for it) rate-ctl is still -0.010 below guarded(hebr-tuned) (Holm LOSE held, -0.0097 Holm p=0.02 LOSE confirm).

### VERIFY REQUEST — W9 — g8_snn: spiking WTA selector + eligibility traces + population coding (rejection)
Rerun (about 10-12 min on 3 threads, deterministic; the run writes both JSONs):
```
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r1 RAYON_NUM_THREADS=3
cd C:/Users/mariu/neural-mesh
cargo build --release --offline --bin g8_snn
$CARGO_TARGET_DIR/release/g8_snn.exe --selfcheck        # 5 sanity checks (WTA picks max drive 400/400, learners learn, pop code peaks)
$CARGO_TARGET_DIR/release/g8_snn.exe --out results/g8s_snn.json --confirm-out results/g8s_snn_confirm.json > results/g8s_snn.log
```
Expected: `held-out: accepted=["rate-ctl (non-spiking)"]` and the same for confirm; there is NO snn-lif / snn-izh entry in `wins`. Defaults: tune seeds 1-8, held 31-60, confirm 61-90, 200 random-search samples per candidate family and per matched-effort baseline family per world.

Protocol: worlds fixed in g8_common/worlds.rs before any policy ran (a short 8-sample smoke run on 6 seeds was run once to test the pipeline; no world was changed afterwards). Per-world tuning on seeds 1-8 for baselines and candidates alike. Baselines: keyword, eps-greedy (9 x attribution grid), pooled-ts, hebbian-robust, warm(hebr) (25 x grid), guarded-safe (e1 default), plus matched-effort random-searched (200 samples) hebbian / warm(hebbian) / guarded(hebbian) over eta, temperature, decay, epsilon, baseline_alpha, row_cap, ucb_c, median-agg, and gate params. Best fair baseline per (world, window) = argmax held-out mean over baseline rows (conservative). Holm over 24 cells = {snn-lif, snn-izh, rate-ctl} x counted worlds {A,B,C,D} x windows {all, late-half}. Bar: Holm win in >1 counted world.

Worlds (6 agents; metric = mean success probability of the chosen agent; oracle/uniform-random printed in the log):
- A cat-delayed-id: 8 topics x 3 keywords, 2 trap topics, drift at n/2 (3 topics change best agent, keyword map goes stale), reward after U{1..20} steps with the decision id. oracle 0.850, random 0.407.
- B cat-anon-delay: 10 topics x 4 keywords, 2 traps, reward after Geometric(1/6) steps WITHOUT id (only reward bit + arrival time); baselines are fed through lag-L or geometric-sampled attribution adapters (L, mean tuned on 1-8; the true mean 6 is in the grid, i.e. generous). oracle 0.850.
- C cont-smooth: x in [0,1]^2, Gaussian competence bumps (0.10 + 0.80 exp(-d^2/(2*0.2^2))). Baselines get coarse 4x4 cell (keyword-map keys), fine 8x8 cell, 8-bin x0, 8-bin x1 tokens. oracle 0.559, random 0.261.
- D cont-ridge: oblique sigmoid half-planes (0.12 + 0.76 sigmoid(10((x-.5).dir - b))). Same tokens. oracle 0.834, random 0.510.
- E static-immediate (control), F cont-noinfo (control: C's probabilities but the learner sees independent random x/tokens). Not counted.

HELD-OUT 31-60, mean expected success (window=all). `-` = n/a.
| router | A | B | C | D | E(ctl) | F(ctl) |
|---|---|---|---|---|---|---|
| keyword | .5758 | .6910 | .4585 | .7297 | .6560 | .2621 |
| eps-greedy | .7118 | .5174 | .4789 | .7830 | .7533 | .2645 |
| pooled-ts | .6721 | .4688 | .4454 | .7674 | .7242 | .2658 |
| hebbian-robust | .7446 | .5195 | .4724 | .7837 | .7753 | .2658 |
| warm(hebr) | .7527 | .6908 | .4861 | .8024 | .7844 | .2649 |
| guarded-safe | .7443 | .6352 | .4935 | .8056 | .7841 | .2658 |
| hebr-tuned (200) | .7450 | .5224 | .4815 | .7845 | .7773 | .2682 |
| warm(hebr-tuned 200) | .7483 | .6907 | .4930 | .7966 | .7785 | .2650 |
| guarded(hebr-tuned 200) | .7567 | .6803 | .5046 | .8115 | .7879 | .2688 |
| snn-lif | .7583 | .5692 | .4867 | .7951 | .7923 | .2661 |
| snn-izh | .7534 | .5314 | .4867 | .7950 | .7678 | .2636 |
| rate-ctl (non-spiking) | .7643 | .5942 | .4948 | .8197 | .8010 | .2664 |

Candidate minus best fair baseline (window=all), paired diff [95% CI], Holm p over 24 cells; confirm seeds 61-90 in the second bracket:
- snn-lif: A +0.0016 [-0.003,+0.006] p=.94 (confirm +0.0007, p=1); B -0.122 [-0.140,-0.103] LOSE (-0.128); C -0.018 [-0.021,-0.015] LOSE (-0.014); D -0.016 [-0.020,-0.013] LOSE (-0.015).
- snn-izh: A -0.003 [-0.007,+0.000] p=.31 (confirm -0.003 p=.75; A-late LOSE p=.044 / .012); B -0.160 LOSE (-0.165); C -0.018 LOSE (-0.017); D -0.017 LOSE (-0.014).
- rate-ctl (non-spiking): A +0.0076 [+0.0033,+0.0120] p=.0116 WIN vs guarded(hebr-tuned) (confirm +0.0084 [+0.0041,+0.0126] p=.0044 WIN); B -0.097 LOSE (-0.086); C -0.0098 [-0.014,-0.006] p=.0003 LOSE (-0.0097 p=.02 LOSE); D all +0.0082 [+0.0049,+0.0116] p=.0003 WIN, D late +0.0057 [+0.0023,+0.0091] p=.016 WIN (confirm all +0.0076 p=.0001, late +0.0047 p=.0093). A-late and C-late n.s.
Wins therefore exist only for the non-spiking control, in A (all) and D (all, late) on both seed sets; none for spiking candidates.

Controls / negatives (all measured):
- E (static, immediate): rate-ctl +0.0131 [+0.008,+0.018] over best baseline, snn-lif +0.0044 [-0.0001,+0.0088], snn-izh -0.020 [-0.025,-0.016]. So rate-ctl's A win is mostly a general "NLMS value-learner beats the Hebbian row rule" effect, not a delay effect.
- F (no information): all candidates within -0.005..+0.000 of the best baseline (nobody can beat the best constant agent: 0.269 vs oracle 0.559), i.e. no spurious gain from the encoder.
- Ablations (row minus parent, window all, held-out): snn-lif -inhibition: A -0.007 [-0.010,-0.004], C -0.004 [-0.007,-0.000], D -0.000, E -0.004 (lateral inhibition helps slightly or is neutral); snn-lif -trace(lambda=0) in B -0.010 [-0.035,+0.016]; rate-ctl -trace in B +0.024 [+0.002,+0.046]; snn-lif[lag-L] in B -0.077 [-0.098,-0.056] (trace > fixed lag); rate-ctl -pop-code: C -0.013 [-0.018,-0.008], D -0.006 [-0.009,-0.004]; rate-ctl -tokens: C -0.008 [-0.013,-0.002], D -0.015 [-0.020,-0.011].
- Spiking vs same-learner softmax (snn-lif - rate-ctl): A -0.006, B -0.025, C -0.008, D -0.025, E -0.009, F -0.000 (all CI-excluding-0 except F); snn-izh - rate-ctl: A -0.011, B -0.063, C -0.008, D -0.025, E -0.033.

Budget match (ns/decision = select+update, single thread, median of 3 held seeds, measured while the shared box was loaded; two runs of the same row differed by up to ~2x, so only order-of-magnitude statements are safe; neurons = 6 agent + input neurons actually used (population + token slots + 1 bias); bytes = exact weights + trace + token table for SNN rows, lower bound distinct-tokens x 6 x 8 B for the HashMap baselines; full table in the JSON `cost`):
| row | A ns | B ns | C ns | D ns | neurons (A/B/C/D) | state bytes (A/B/C/D) |
|---|---|---|---|---|---|---|
| warm(hebr) | 1213 | 1141 | 3181 | 1903 | 0 (rows) | >=1152/1920/4608/4608 |
| guarded(hebr-tuned) | 1867 | 1102 | 2755 | 1424 | 0 | same |
| snn-lif | 1332 | 3912 | 3335 | 5348 | 6+25 / 6+41 / 6+133 / 6+53 | 5136/10112/8256/6688 |
| snn-izh | 495 | 3700 | 4808 | 1011 | 6+25 / 6+41 / 6+37 / 6+121 | 5136/10112/6384/7680 |
| rate-ctl | 282 | 2207 | 782 | 962 | 6+25 / 6+41 / 6+33 / 6+81 | 5136/10112/5728/8032 |
The spiking selectors are not cheaper than the Hebbian routers on a CPU simulation (LIF 1.3-5.3 us vs 1.1-3.2 us; Izhikevich is cheap only where first-spike stopping fires early, A and D); rate-ctl is similar or cheaper but holds ~1.5-5x more state than the lower-bound weights of the baselines. Spike diagnostics: LIF winners fire 1.00 of decisions, 12-77 spikes/decision; Izhikevich 0.94-1.00 of decisions, 1-8 spikes/decision (no sparse-event energy argument applies on a CPU simulation).

Native implementation summary (for reviewers): LIF (leak .8, thr 1, reset 0) and Izhikevich (RS/FS presets, two dt/2 Euler substeps, spike at 30 mV) neurons, one per candidate agent; input spikes are Bernoulli draws with p_j = min(1, lambda_in * r_j) from Gaussian tuning curves (separate 1-D codes or a 2-D grid) plus one-hot keyword/token neurons plus a bias neuron; lateral inhibition subtracts `inh` from all other candidate neurons per spike; winner = max spike count, ties by first-spike latency then potential; optional first-spike stop; uniform membrane noise. Learning = three-factor rule with a normalised-LMS reward-prediction error (pre = normalised input rate, post = chosen agent, modulator = r - yhat), eligibility trace E (decay lambda per decision) for anonymous reward, or exact snapshot for id'd reward, or lag-L attribution (control). Oxicuda-snn/neuromod sources were read only for conventions; nothing copied or built.

Caveats the director should weigh: (1) candidates' 200-sample random search covers a different, larger space than the baselines' grids (matched-effort Hebbian rows were added to counter this; the rate-ctl margin shrank from +0.012/+0.014 to +0.008/+0.008 when they were added, i.e. the conclusion is sensitive to baseline tuning effort); (2) everything tuned per world on 8 seeds; a single cross-world config was not tested; (3) all four counted worlds are mine (A and B share a generator family; C and D share the token set), so "independent worlds" is generous; (4) B is a world where no learner beats the keyword rule, so it does not test whether traces help in principle, only that they do not rescue learning here; (5) timings are single-run on a loaded box; (6) the Izhikevich/LIF parameter ranges are one hand-picked family, a wider search could change the SNN numbers but the spiking-minus-rate gap (-0.006 to -0.025, tight CIs) makes a reversal unlikely for this mechanism.

History (disclosure): v1 run (no matched-effort baselines): rate-ctl won A (+0.0116 held / +0.0085 confirm, Holm .003/.048) and D (+0.014 / +0.011) versus the original baselines; snn-lif/izh no wins. v2 added matched-effort baselines (same verdict, smaller margins). Final run added the rate-ctl ablation rows only (no change to any other number). The SNN verdict never changed.
