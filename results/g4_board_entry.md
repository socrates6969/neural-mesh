# G4 (W4, round 3) -- creativity / diversity routing, revisited honestly

Scope of every claim below: ROUTING of tasks to agents under a diversity-valued objective, in my own simulated worlds. Nothing here says anything about
real model output, real novelty, "creativity" of an AI, or real traffic. All worlds, the value function and the policies are mine (see "limits").

Files (only these were added): `src/bin/g4_creativity.rs`, `src/bin/g4_robust.rs`, `src/bin/g4_common/{mod.rs,world.rs,world_alt.rs,pols.rs}`, `results/g4_*`.
`src/*.rs` untouched; no new crates; no lib patch needed (nothing proposed in `results/g4_proposed_patch.md`). No git commits.

COMMON RERUN PREAMBLE (each run ~10 s on 3 threads; build ~15 s):
```
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r4; export RAYON_NUM_THREADS=3; cd /c/Users/mariu/neural-mesh
sha256sum src/bin/g4_common/world.rs   # must equal ea287f5903554394f24bd3e47a2061cac7b84d964d4363c843d2188ecfb36224 (value function frozen before any policy ran; results/g4_world_frozen.md)
```

## What was built (and fixed in advance)
* Value function frozen first (`results/g4_world_frozen.md`, sha256 above, timestamp 2026-10-04T06:37Z, before any policy existed). Creative domain value = q*(0.25+0.75*nov),
  nov = nearest-neighbour distance of the new output embedding (4-D, hidden agent style + noise) to the last 8 outputs of that domain; correctness domain value = q only (repeats fine).
  Agents have hidden per-domain quality q and hidden style vectors. The router's reward is the scalar value.
* 9 worlds (1200 tasks, 8 agents): C1 multistyle, C2 one-star-plus-diverse, C3 clustered styles + quality correlated with style + style drift @600 [creative];
  K1/K2/K3 same structures with value = quality only [controls]; M1 (3/6 creative), M2 (1/6 creative, star) [mixed domains, shared router]; R1 "rigged check" (see below).
* Fair baselines = task's list {keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebr), guarded-safe (= defaults::recommended_router)}, each tuned PER WORLD on seeds 1-8 (generous),
  best of them per world = [G]. Also [P] = same baselines tuned once pooled over worlds (same protocol as arms), [U] = plain domain-keyed UCB (my own learner, no bonus; NOT in the task's list; an extra-strong reference), [L] = the arm's own base without the diversity layer.
* Diversity arms (all wrap the public `Router`/types inside my bin; hyper-parameters tuned ONCE pooled over all 9 worlds on seeds 1-8 -- a deployer does not know which world they are in):
  anti-repeat (id-only candidate mask, DPP-lite), novfilter (shortlist top-k agents by the policy's OWN novelty estimate, base = warm(hebbian-robust)), archive (MAP-Elites-flavoured cell staleness shortlist),
  nbucb (own UCB + additive novelty bonus beta*nov_est), nbreg (post-hoc, untuned, regression-calibrated novelty, see below).
  Gates: none | declared (ORACLE-ISH: told which domains are creative) | learned (per domain, discounted within-agent regression of reward on the policy's own observed novelty, on if t>2.5 & slope>.15, off if t<1; start OFF).
  Policies see: features, scalar reward, the output embedding (obs dims). Policies do NOT see Q, true styles, or the value formula. Their novelty model (mean of 3 nearest, window 10, scale 1) differs from the world's (min distance, window 8, scale 1).
  An ORACLE-ISH ceiling arm (knows true styles/exact value form/which domains are creative; learns only q) is in the logs as a ceiling, not a candidate.
* Protocol: tune seeds 1-8; held-out 31-60 (`--set held`); fresh confirm 61-90 (`--set confirm`), same frozen/deterministically re-derived configs; paired 95% t-CI; Holm per test family (win+harm vs G; vs P; vs U; vs L);
  harmlessness = TOST equivalence at delta 0.01 (reported uncorrected and Holm) and non-inferiority (90% lower bound > -0.01).

## PROCESS DISCLOSURE (read before trusting any number)
1. First full run (`results/g4_creativity_held_firstrun_EXPLORATORY.log`) exposed a bug in my REFERENCE rows only: the "base-alone" reference of the nbucb arms was a pooled-TS router instead of nbucb with beta=0, and I had no pooled-tuned/UCB reference. I fixed the references and re-ran; the arms and their configs did not change (arm blocks are numerically identical apart from Holm-family sizes). So held-out 31-60 was SEEN once before the final run; only the confirm set 61-90 is clean from that. No arm was tuned or altered because of what I saw there.
2. The robustness variants V1/V2 were run after the main results. The `nbreg` arm was added AFTER seeing that nbucb fails V2: it is a post-hoc design response, has no tuned hyper-parameter (c=0.05 copied from the nbucb grid, clip 1.5, same gate constants), was sanity-checked on dev seeds 101-130 (`results/g4_creativity_dev101-130_nbreg_sanity.log`), then evaluated on held/confirm. V3 (floor .35, window 6, scale .8) was defined after V1/V2 and after nbreg existed, before it was run on V3, so V3 is the only clean out-of-sample misspecification check for nbreg; V2 is not (design was informed by it).
3. `RunOut`/`View` are the only channels; grep `v\.` in `pols.rs` shows only `Oracle` reads the world view.

---

### VERIFY REQUEST — W4 — G4-A: gated novelty-bonus bandit beats the best fair baseline on all 3 creative worlds, harmlessly, INSIDE my worlds
Claim (narrow): when value rewards novelty vs the recent outputs of a domain and the policy can observe a (partially) matching output embedding, a bandit that adds a novelty term for creative domains only routes better than the best fair router, and with a gate it costs nothing on correctness domains.
Rerun:
```
cargo run --release --offline --bin g4_creativity -- --set held --out results/g4_creativity_held.json > results/g4_creativity_held.log
cargo run --release --offline --bin g4_creativity -- --set confirm --out results/g4_creativity_confirm61-90.json > results/g4_creativity_confirm61-90.log
grep -A40 "^## nbucb(b=0.8,c=0.05)|learned" results/g4_creativity_held.log      # same for "nbreg(c=0.05)|learned", "nbucb(b=0.4,c=0.05)|declared"
```
Numbers (paired mean diff in creative-domain value [95% CI], 30 seeds, seed wins 30/0 or 28-29/1-2; Holm-significant everywhere listed):
* nbucb|learned (b=.8,c=.05) vs [G] best fair baseline: held C1 +0.0716 [+.066,+.077]; C2 +0.0633 [+.058,+.069]; C3 +0.0287 [+.023,+.034]; M1 creative half +0.0991; M2 creative half +0.0926. confirm: C1 +0.0770, C2 +0.0690, C3 +0.0267 (28/2), M1 +0.1011, M2 +0.1070.
* nbreg|learned (untuned) vs [G]: held C1 +0.0693, C2 +0.0569, C3 +0.0263, M1 +0.0990, M2 +0.0884; confirm C1 +0.0768, C2 +0.0574, C3 +0.0237, M1 +0.1032, M2 +0.1032. nbucb|declared (b=.4) vs [G]: held +0.0804/+0.0697/+0.0253; confirm +0.0847/+0.0733/+0.0250.
* Layer isolation [L] (arm minus its own base without the novelty term; this is the causal effect): nbucb|learned held C1 +0.0794, C2 +0.0643, C3 +0.0262, M1 +0.0674, M2 +0.0535; nbreg|learned held +0.0486/+0.0348/+0.0131/+0.0452/+0.0329 (its base is a stronger plain-UCB-EMA so the layer adds less).
  The creative-world win vs G is therefore essentially the novelty layer (plain domain-keyed UCB [U] alone vs G on creative value: C1 -0.008 held/-0.012 confirm, C2 -0.001/-0.005, C3 +0.003/+0.004, i.e. NOT the source of the creative win).
* Share of headroom captured (ORACLE-ISH ceiling minus G): C1 ~59%, C2 ~55%, C3 ~44% (held, nbreg/nbucb learned). Oracle: C1 .516, C2 .469, C3 .345 vs G .399/.365/.285.
* Harmless (control worlds K1,K2,K3 and the correctness half of M1,M2; delta 0.01):
  vs [L] (own base): nbucb|learned diff -0.0017/-0.0007/-0.0011/-0.0009/-0.0002 held (TOST equivalence 5/5, Holm 5/5), confirm 5/5 (K2, M1, M2 within 0.0000 because the gate stayed off). nbreg|learned 5/5 equivalent (Holm 5/5) held and confirm. nbucb|declared 5/5 (exactly 0.0000 by construction: declared gate = pass-through).
  vs [G] (the task's comparison): non-inferior 5/5 (no significant loss anywhere); strict TOST-equivalence fails 4/5 only because the arm is BETTER than the fair baselines there (K1 +0.011, K2 +0.024, M1 +0.015, M2 +0.028 -- that is the plain-UCB learner being stronger in worlds with confound features, not a diversity effect).
* Learned-gate diagnostics (held): gate on in 82-83% of creative-domain tasks (starts OFF, ~17% burn-in), 0.00-0.02 of correctness-domain tasks; one false-positive pocket: nbreg|learned in K3 gated on 10% of tasks (style-correlated quality), no measurable harm (-0.0001).
Cost of learning the gate vs being told: nbucb learned vs declared in C1 +0.0794 vs +0.0882 [L], i.e. ~0.009 lost to burn-in/threshold.
Negatives inside this claim: (1) R1 below. (2) vs [U]/[L] in R1 the bonus costs -0.004..-0.009 (Holm-significant on confirm only). (3) the headroom captured is only ~45-60% of the ceiling.
Verdict I believe: PASSES the pre-registered bar (wins >1 independent creative world with Holm on held AND confirm; harmless on all 5 correctness cases, TOST delta .01) -- but only for the gated novelty-bonus bandit (nbucb/nbreg, declared or learned gate), only in worlds whose value function I wrote, and see G4-D for the robustness failure.

### VERIFY REQUEST — W4 — G4-B: three of the four diversity mechanisms FAIL; ungated novelty hurts correctness badly
Rerun: same commands as G4-A; `grep "^## \|SUMMARY" results/g4_creativity_held.log | cut -c1-260`.
* anti-repeat (id-only mask, DPP-lite), declared gate, vs [G] creative: held C1 -0.0256, C2 -0.0183, C3 -0.0053, M1 -0.0061 (all Holm-significant LOSSES); only M2 (+0.008, Holm .23, ns). Learned gate: C1 -0.0291, C2 -0.0179, C3 -0.0042.
  Reason (my reading): excluding the last m agents sacrifices quality without knowing which excluded agents were stylistically distinct; the bandit already rotates somewhat on its own through reward feedback.
* archive (MAP-Elites-flavoured sign-pattern cells, staleness shortlist) declared: C1 -0.0230, C2 -0.0178, C3 -0.0082, R1 -0.0041; M2 +0.0116 only. It is "MAP-Elites-flavoured", not a faithful quality-diversity archive; I do not claim pyribs-level coverage of that idea.
* novfilter (hard shortlist, k=2, base warm(hebbian-robust)) WORKS with a gate (declared vs G: +0.0514/+0.0462/+0.0165 held; learned +0.0354/+0.0346/+0.0146), but: in mixed worlds the shared base router is contaminated by the restricted creative-domain updates: learned novfilter M1 correctness half -0.0094 [-0.0127,-0.0061] vs own base (fails equivalence, p .36), declared -0.0105; this is a real cross-domain cost of sharing one router, absent in nbucb/nbreg (per-domain statistics).
* UNGATED (gate=none, pooled-tuned compromise params): novfilter(k=6) correctness vs own base K1 -0.0417, K2 -0.1547, K3 -0.0617, M1 -0.0479, M2 -0.1360 (all significant); nbucb(b=.1) K1 -0.0175, K3 -0.0098, M1 -0.0154 (significant, 3 of 5 fail equivalence). Applying novelty to every domain is harmful (consistent with the R2 finding); the gate is what makes the idea acceptable.
Verdict I believe: do not pursue anti-repeat or archive routing as routing mechanisms in these worlds; novfilter only with a gate and per-domain routers.

### VERIFY REQUEST — W4 — G4-C: the gate must be learnable; learned gate works here but is a weak point
Rerun: `grep "gateon" results/g4_creativity_held.log | cut -c1-200`.
* Learned gate = within-agent regression of reward on the policy's own observed novelty (demeaned per (domain,agent), discount .995). It switched on in creative domains and stayed off in correctness domains in 8/9 worlds including the style-correlated-quality control (K3: 0-10%). It works because reward-vs-own-novelty covariance exists even for a single exploited agent (output noise changes the distance to its own recent outputs).
* Fragility: (a) it measures an OBSERVATIONAL slope; it cannot tell whether switching agents would buy novelty (see V2 below). (b) thresholds (t_on 2.5, slope .15) are fixed constants, not tuned; not stress-tested. (c) in R1 it also turns on (see next block).

### VERIFY REQUEST — W4 — G4-D: robustness / "rigged" checks -- THE IMPORTANT NEGATIVE
Rerun (each ~8 s):
```
for v in v1 v2 v3; do for s in held confirm; do cargo run --release --offline --bin g4_robust -- --variant $v --set $s > results/g4_robust_${v}_${s}.log; done; done
awk '/^## /{arm=$2} /win(G|U) +[CM][123]-/{print arm, $1, $2, $3}' results/g4_robust_v2_held.log
```
Value-function variants the policies were NOT built for (arms frozen from the main run; fair baselines re-tuned per world = generous to baselines):
V1 floor .5, window 5, scale 1.2, MEAN-distance novelty; V2 floor .1, window 12, scale .6, min-distance; V3 floor .35, window 6, scale .8 (clean OOS for nbreg).
* nbucb|declared (b=.4): V1 beats G in C1-3 (+0.020/+0.021/+0.019 held, +0.020/+0.018/+0.016 confirm); nbucb|learned (b=.8) does NOT (V1 C1 +0.001 ns, C2 -0.008 sig loss, C3 -0.003 ns) -- the pooled-tuned strength b=.8 over-chases when novelty is worth less. novfilter loses to G in V1 in all C worlds (up to -0.036).
* V2 (novelty practically unattainable: window 12 > #agents 8 so every agent's previous output is always in the window; optimal = pure quality): EVERY arm loses to the best fair baseline in ALL creative worlds, Holm-significant, both sets: nbucb|declared -0.060/-0.054/-0.027 (held C1-3), nbucb|learned -0.051/-0.045/-0.027, novfilter -0.039..-0.023, nbreg -0.049/-0.041/-0.019; vs plain UCB -0.02..-0.06 (nbucb -0.03..-0.06, nbreg -0.02..-0.05). The gate stays on because within-agent reward covaries with observed novelty (observational slope) while actually moving to a "novel" agent gains little. A deployed system would pay ~0.02-0.06 of value for chasing novelty that does not exist.
* V3 (milder mismatch): nbreg (declared and learned) beats G in C1-3 and M1,M2: held +0.0585/+0.0523/+0.0284/+0.0893/+0.0838, confirm +0.0607/+0.0541/+0.0293/+0.0982/+0.0967 (nbreg|learned within 0.001). V1 nbreg: C1 +0.016..+0.020 (sig), C2 +0.001..+0.003 (NOT significant, Holm 1.0), C3 +0.007..+0.010 (sig), M1 +0.031..+0.033 (sig), M2 -0.007..+0.005 (ns/mixed).
* R1 (rigged check; value novelty on hidden dims 2-3, policies observe dims 0-1 only): I intended "uninformative observation" but that is not exactly true -- repeating the same agent shows up in BOTH the observed and the hidden dims, so the observation still reveals repeats and the learned gate switches on (~80%). Result: arms beat the best fair baseline by a small +0.003..+0.014 (nbucb|learned +0.0121 held, +0.0067 confirm; novfilter +0.003/+0.0005) but are 0.004-0.009 WORSE than their own base without the bonus (vs [L]: nbucb|learned -0.0053 held (Holm .125, ns), -0.0086 confirm (Holm-significant); nbreg -0.0040 held (ns) / -0.0045 confirm (Holm .003)): the bonus itself gives nothing there; the vs-G "win" is the stronger learner. This is why [L]/[U] are reported next to [G].
Verdict I believe: the positive result is conditional on the policy's internal novelty scale/window roughly matching the world's; there is no mechanism that detects "novelty is not purchasable here". Fails V2 completely; passes V3 (nbreg), partially V1.

---
## Where it helps / where it does not (one table)
| situation | result |
|---|---|
| novelty valued, observation matches, gate right (C1-C3, creative halves of M1/M2) | +0.024..+0.11 value vs best fair baseline (Holm, 30/0 wins), ~45-60% of ORACLE-ISH headroom |
| correctness-only domains with declared/learned gate | exactly or practically equal to the same learner without the layer (TOST .01 5/5) |
| ungated novelty (all domains) | -0.002..-0.15 on correctness value; never use |
| anti-repeat, archive mechanisms | lose in creative worlds even with the right gate |
| shared single router across domains + hard shortlist (novfilter) | -0.009..-0.011 on the correctness half of M1 |
| observation only partially informative (R1) | bonus worth nothing (-0.004..-0.009 vs own base); win vs G is learner quality |
| novelty not purchasable / scale mismatch (V2) | all arms lose 0.02-0.06 vs best fair baseline; gate does not protect |
| mild mismatch (V3), moderate (V1) | nbreg still wins (V3 clearly, V1 small); nbucb|learned loses in V1 C2 |

## Honest limits
* Worlds, value function and policies are all mine; the value function was frozen first, but "novelty vs recent outputs measured on an embedding the policy also observes" is exactly the structure that favours the idea. No real outputs, no real embeddings, no human judgment of novelty.
* 4-D synthetic styles; 8 agents; one task stream design; single noise level. 30 seeds per set; seeds differ in world draw AND noise (a world-level effect would show as a world-wise split I did not run).
* The gate constants and the policy's novelty definition were fixed a priori but are not independent of my world design (I wrote both). nbreg is post-hoc (see disclosure).
* held-out 31-60 was viewed once before the final run (reference-row bug fix only); confirm 61-90 and dev 101-130 seeds were not used for design except as stated.
* "Equivalence/non-inferiority" are at delta 0.01 absolute value (about 1.2-2.5% of the value level); I also print delta .02.
* I did not do a pyribs cross-check or a faithful MAP-Elites; "archive" is a cell-staleness heuristic.
* The comparison with the task's baselines mixes learner quality and novelty layer; [L] is the clean causal comparison and is what I would quote.

## Recommendation
Not a default and not a merge candidate. A gated, per-domain novelty-bonus bandit is a defensible opt-in for domains that a human declares creative AND where an output-embedding novelty measure that matches the intended value exists; in that case the evidence supports "+0.02..+0.10 value in my simulated worlds, no cost elsewhere". Do not enable learned gating blind: V2 shows it can pay 0.02-0.06 for nothing. No claim about real creativity, speed, or knowledge is made.
