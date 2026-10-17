# H7 (R4-C1) -- creative opt-in win, widened honestly: baseline best-of-k first, verbalized sampling, new generators

LABEL: every number is SYNTHETIC-until-validated-on-real-traffic. No LLM was called. Scope = routing / candidate selection in simulated worlds I (or W4) wrote.
No claim about real creativity, novelty or model quality. Only files added: `src/bin/h7_lab.rs`, `src/bin/h7_common/{mod,hworld,engine,pols2}.rs`,
`results/h7_*` (logs, json, `h7_summarize.py`, `h7_run_all.sh`, `h7_worlds_frozen.md`, `h7_proposed_patch.md`). `src/*.rs` untouched, no crates, no commits.
g4 files reused through `#[path]` unmodified (`world.rs` sha256 ea287f59... verified at the end; `pols.rs` unchanged).

REBUILD / RERUN (~5 min per set; box-safe: 2 threads):
```
export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r7 RAYON_NUM_THREADS=2; cd /c/Users/mariu/neural-mesh
cargo build --release --offline --bin h7_lab
sha256sum src/bin/g4_common/world.rs src/bin/h7_common/hworld.rs   # ea287f59... / 80e2071c...  (results/h7_worlds_frozen.md)
target=C:/Users/mariu/nm-target-r7/release/h7_lab.exe
$target --exp validate                                   # engine k=1 == g4 run(), max |diff| 0.000e0 over 8 W4 worlds x 10 seeds x 7 policies
bash results/h7_run_all.sh                               # every table below: exp in {bok,vs,gate} x worlds in {w4,n,w4v2,w4v3,w4v4,tag} x set in {held,confirm}
python results/h7_summarize.py bok w4 "G (best" "BoK-dedup k=3"   # compact tables (arg3 = reference arm, then arm-name substrings)
```
Protocol = common rules: fair baselines [G] (the g4 fair list/grid) tuned PER WORLD on seeds 1-8 (generous); every H7 selection config tuned ONCE pooled
over the world set on seeds 1-8 and frozen; held-out 31-60, fresh confirm 61-90 (same deterministic config); paired 95% t-CI, Holm per reference family,
TOST +-0.01 and 90%-lower-bound non-inferiority for correctness metrics. 30 seeds per cell.

## WHAT WAS BUILT
* Engine (`h7_common/engine.rs`): multi-candidate runs over the frozen W4 generator and 4 new generators, common random numbers (slot 0 = the native single call),
  costs counted per creative-valued task: generator-call equivalents + judge/verifier calls. A creative-TAGGED task (human tag, "declared") triggers k candidates; others take the
  normal single call. Only the delivered candidate updates the router and the novelty state.
* Candidate selectors: `QOnly` (judge's best quality = cost-matched baseline), `Dedup{theta,w}` (drop candidates within euclid theta of any of the last 10 delivered outputs
  -- for unit-norm embeddings cos = 1 - theta^2/2 -- then argmax qhat*((1-w)+w*nov_hat), nov_hat = the policy's own mean-of-3-NN distance), `Arch` (archive staleness of the
  candidate's own cell, W4-style sign cells or a designer-given 8x8 grid), and a CLAIRVOYANT `OracleVal` (true value; ceiling only, not a method).
  Judge: qhat = realized q + N(0,.05) (a priori default; sensitivity .0/.1/.2/none in the logs).
* Verbalized sampling (VS): ONE router pick, ONE call that returns k=4 candidates: candidate 0 = the native sample, candidates 1..3 = native + offset ~ N(0, (sigma*scale)^2 I)
  with quality penalty kappa*||offset||/scale. ASSUMED profile (pre-declared before any VS run): sigma .35, kappa .15, call price 1+rho*(k-1) with rho .3, plus the judge calls.
  Sensitivity: sigma {.15,.35,.6} x kappa {0,.15,.4,.8,1.5}; rho {0,.3,1}; judge-free variants.
* New generators (frozen in `hworld.rs` BEFORE any H7 policy ran; hash in `h7_worlds_frozen.md`): N1 grid/QD archive (permanent 8x8 archive = fixed coverage objective, per-cell
  quality landscape), N2 global corpus (novelty vs 60 seed outputs + EVERY delivered output of all domains, kNN, no window), N3 modal agents (2 modes each) with CONCAVE novelty utility
  (diminishing returns), N4 grid with a 24-cell recent-memory archive (novelty practically unattainable BY ROUTING), NK1-3 value=q controls, NM1/NM2 mixed 3-of-6. Plus V2/V3/V4 value
  variants on the W4 generator (V2/V3 = W4's own; V4 = floor .15 / window 16 / scale .5, defined before any DX arm ran).
* New gate modes for the W4 nbucb/nbreg learners (`pols2.rs`, reimplementation validated bit-exact vs g4 NbUcb/NbReg): RX (counterfactual: epsilon=.1 random-agent experiments, regress reward on the
  PRE-selection predicted novelty advantage), AB (periodic gate-off, two-armed bandit), DX (randomised on/off picks only on tasks where the two picks disagree), DXT (DX + Welch t>=2 evidence required, default OFF).

## PROCESS DISCLOSURE (read before trusting a number)
1. `exp validate` (exact reproduction of g4 `run`) was run before any experiment. `hworld.rs` was edited once after freezing: `gen`->`gk` (edition-2024 keyword; before any policy ran; new hash recorded).
2. First batch (bok/vs/gate, held+confirm) was run, I READ THE HELD LOGS ONLY. Then, POST-HOC and in response to what I saw: STACK arms (nbucb+dedup); oracle rows; VS kappa {.8,1.5}, judge-free and rho arms;
   the RX/AB arms were pre-declared, DX and DXT are post-hoc responses to RX/AB not fixing V2 (iterated on DEV seeds 101-130, dev logs deleted). Everything was then re-run from scratch (all
   configs re-derived deterministically) on held and confirm; confirm 61-90 was not read before the final batch. V2 is NOT out-of-sample for the gate designs (W4 + my RX/AB results informed them);
   V3 (W4's) and V4 (mine, fixed before DX ran) are the less-contaminated checks.
3. Honest bias in my favour that I could not remove: dedup/archive selectors and W4's value function share the same "distance to recent outputs" structure; N1-N4 are also my worlds. The judge sees the
   realized quality (+noise) of every candidate, i.e. best-of-k gets a verifier for free in quality terms; I charge it as calls.
4. VS numbers are properties of the ASSUMED profile, not of any model. I show how they move; I do not claim any real LLM has such a profile.

---
### VERIFY REQUEST -- H7 -- H7-A: best-of-k + embedding dedup is the honest baseline; it wins everywhere but is bought with calls, and the diversity filter generalises only partly
Rerun: `h7_lab --exp bok --worlds {w4|n|w4v2|w4v4} --set {held|confirm}`; logs `results/h7_bok_<w>_<set>.log` (SUMMARY block at the end), json `results/h7_bok_<w>_<set>.json`.
Tuned (pooled seeds 1-8, frozen): dedup k=3 W4-type worlds theta .15 w .75; new generators theta 0 w .5 (NOT the same -> not portable).
Creative-value diff vs [G] best fair routing (1 call), held | confirm (all Holm-significant unless marked ns):
* BoK-dedup k=3 (3 gen + 3 judge calls = +5 extra calls per creative task): W4 C1 +.0489|+.0511, C2 +.0470|+.0487, C3 +.0345|+.0332, M1 +.0630|+.0668, M2 +.0373|+.0351.
  New generators N1 +.0291|+.0269, N2 +.0105|+.0102, N3 +.0645|+.0639, N4 +.0503|+.0483, NM1 +.0450|+.0431, NM2 +.0165|+.0155. V2 +.029/.030/.036, V4 +.025/.032/.032 (held; confirm same).
  k=4: W4 +.0634/.0581/.0365; k=2: +.0347/.0320/.0205. Judge-free dedup k=3 (only +2 extra calls): W4 +.0398/.0357/.0217 | +.0420/.0384/.0198 but N1 +.0015 ns, N2 -.0057 LOSS, V2 weak -> the judge matters outside W4.
* Cost: break-even price = gain per extra call = ~.0098 value/call (C1, calls priced equal, 5 extra) i.e. ~2.5% of the G value level per call; judge-free ~.02/call. A gain that needs 2-3x the calls is a loss unless a call costs < that.
* The COST-MATCHED baseline is quality-only best-of-3 (BoK-q): W4 +.0168/+.0225/+.0178 | +.0156/.0206/.0184; N1 +.0266, N2 +.0146, N3 +.0403, N4 +.0349. So ~35-50% of the dedup gain in W4 and most of it in N1/N2 is
  plain best-of-k with a verifier, not diversity. The DIVERSITY FILTER's own increment (dedup minus BoK-q, same calls): W4 C1 +.0321|+.0356, C2 +.0245|+.0281, C3 +.0166|+.0148, M1 +.0259|+.0313, M2 +.0159 | +.0098 ns;
  V2 +.015/.013/.019, V4 +.004/.013/.014; N3 +.0242|+.0250, N4 +.0154|+.0107; but N1 +.0026 | +.0013 ns, N2 -.0041 | -.0043 LOSS, NM2 -.0136 | -.0127 LOSS.  => the filter does NOT widen to the corpus-novelty generator.
* Ceiling (clairvoyant best-of-8 by TRUE value, 8 calls): W4 +.13/.12/.09, N1 +.057, N2 +.044, N3 +.163, N4 +.264. BoK-dedup k=3 captures 37-39% of it in C1/C2 (k=3 clairvoyant: +.078/+.072 -> 63%), but 19-51% in N1-N4 (N1 51%, N2 24%, N3 40%, N4 19%); N1/N2 have little headroom at all.
* Versus W4's nbucb|learned (b=.8; 0 extra calls) in W4 worlds: BoK-dedup k=3 is WORSE in C1 -.0228|-.0259, C2 -.0163|-.0203, M1 -.036|-.034, M2 -.055|-.072 (all Holm), ties C3 (+.006 ns); k=4 still loses C1, M1, M2 (C2 ns). In the NEW worlds the order flips: BoK-dedup k=3 beats nbucb|learned by +.033/.017/.067/.048 (N1-N4, both sets) and by +.080/.076/.063 in V2.
  STACK (nbucb|declared + dedup k=2/3) adds only +.005..+.016 over nbucb|learned in W4 (Holm in 3-4 of 5 cells) and is NEGATIVE in V2 (-.026/-.020 vs G, k=2) and N2 (-.011).
* Archive/MAP-Elites-flavoured selection (expected to tie dedup): it does NOT tie. W4 k=3: +.0225/.0287/.0207 vs dedup +.0489/.0470/.0345 (loses .014-.026); N3 +.0384 vs +.0645, V2/V4 below BoK-q (Holm loss) in 2 of 3 creative worlds each;
  it ties only in N1/N2 (within .006) and WINS in N4 (+.0760 vs +.0503; confirm +.0750 vs +.0483) where I gave it the world's own 8x8 behaviour grid (designer-known descriptors: favours the archive; with sign cells it loses N4).
* Same-agent native sampling k=3 + dedup (no router diversity, same k calls): +.039/.037/.032 W4, +.033/.024/.061/.047 N1-N4 and beats BoK-q in every pure-creative world incl. N1/N2/V2/V4 (+.006..+.032): the multi-agent variant's 2nd/3rd candidates are lower-quality agents.
* Harmless on correctness: tag-gated arms are IDENTICAL to G on pure-correctness worlds BY CONSTRUCTION (diff exactly 0.0000, flagged "identical by construction" in the logs; this is not independent evidence).
  Mixed worlds (shared router across domains): correctness half of M1/NM1 moves -.004..-.009 vs G (BoK-dedup k=3 held M1 -.0082 [-.0137,-.0026] fails non-inferiority, confirm -.0047 passes; BoK-q -.007/-.007 fails both) -> small cross-domain cost
  of sharing one router while the creative half trains it on delivered-only feedback. With a learned gate instead of tags: 5/5 equivalent.
Verdict I believe: PASSES as an honest baseline (wins in W4-type worlds, 4 new-generator creative worlds and 2 value misspecifications, both sets) but it is a COST trade (3x calls) and its diversity component is specific to generators whose novelty is "distance to recent outputs".

### VERIFY REQUEST -- H7 -- H7-B: verbalized sampling (one call, k candidates) -- big gains that are entirely a function of an ASSUMED profile; non-inferior on correctness only with correct tags or a learned gate
Rerun: `h7_lab --exp vs --worlds {w4|n|w4v2|w4v4|tag} --set {held|confirm}`; the 30-arm profile grid is in the logs (SUMMARY block).
* Default assumed profile (sigma .35, kappa .15, rho .3, k=4, judge on, theta .5 w 1.0 tuned on seeds 1-8): creative diff vs G: W4 C1 +.182|+.179, C2 +.174|+.171, C3 +.199|+.189, M1 +.221, M2 +.235; N1 +.031, N2 +.176, N3 +.101, N4 +.064; V2 +.28..+.30, V4 +.28..+.37. Cost +4.9 extra call-equivalents (.9 gen + 4 judge). Judge-free: +.159/.151/.185 W4 (cost +.9).
  READ THIS CORRECTLY: with offsets of norm ~.7 the candidate is simply far from the recent outputs, so novelty saturates by construction; the size of the gain is the profile, not a finding about models.
* Sensitivity (W4, held|confirm, C1): kappa 0: +.27; .15: +.18; .4: +.071|+.071 (sigma .35), +.055 (sigma .15), +.045 (sigma .6); kappa .8: -.055 (sigma .35), -.012 (.15), -.100 (.6) LOSS; kappa 1.5: -.15 LOSS. Break-even quality penalty ~ kappa .5-.6 per unit embedding distance.
  The selector (tuned for the default) does not adapt, so a worse-than-assumed profile is a net loss; judge-free is far more fragile (N1/N3/N4: kappa .4 -.06..-.09, kappa .8 -.14..-.27, kappa 1.5 -.20..-.41).
  In N-worlds the gain collapses for kappa>=.8 (<=+.01; clamping of offsets at the unit-square edge is my explanation, untested). rho only changes cost (judge calls dominate: 4 of 4.9).
* NON-INFERIORITY on correctness worlds (TOST +-0.01 vs G, 9 worlds: K1-K3 and NK1-NK3 with ALL tasks tagged creative, K1-K3 with 30% false-positive tags; json `results/h7_vs_tag_{held,confirm}.json`):
  default VS with the (wrong) tags FAILS: -.028/-.027/-.024 (K1-K3 all-tagged), -.015/-.013/-.014 (30% FP), -.037 (NK2), -.009 (NK3; NK1 +.039 because the judge improves quality in the landscape world); 0/9 equivalent, 1/9 non-inferior (held; confirm identical).
  VS|learned gate (W4's learned gate, no tags; judge on or judge-free): |diff| <= .0008 in 9/9 worlds, TOST equivalent 9/9, Holm; the gate never switched on there. With CORRECT tags VS is identical to G on correctness worlds by construction (trivial).
  Learned-gate VS keeps ~80% of the creative gain in W4 (+.149/.141/.155 vs +.182/.174/.199 tagged) and 43-81% in N1-N4 (N1 61%, N2 43%, N3 81%, N4 64%).
Verdict I believe: VS is not a finding about real models. Conditional on a profile whose quality penalty is below ~.5/unit-distance it is a large gain at ~1-5 extra call-equivalents; above that it loses. Gate it with the learned gate, never with unvetted tags.

### VERIFY REQUEST -- H7 -- H7-C: the W4 nbucb/nbreg win does NOT widen to new generators; no gate fix removes the V2 harm
Rerun: `h7_lab --exp gate --worlds {w4|w4v2|w4v3|w4v4|n} --set {held|confirm}`; plus the `bok` logs (they contain the frozen W4 arms in every world set).
* W4 arms frozen from g4 (b=.8/c=.05 learned; nbreg learned; b=.4 declared), NOT re-tuned. vs G, held | confirm: N1 nbucb|learned -.0040! | -.0060!, nbreg -.0027 | -.0046!; N2 -.0063! | -.0065!, -.0102! | -.0100!; N3 -.0026 | -.0013, -.0022 | -.0002; N4 +.0028 | +.0016 ns, +.0027 | 0.0000 ns; NM1 (creative half) +.011 Holm, NM2 +.009 ns.
  Against plain UCB [U] the layer is: N3 +.0347 | +.0324 (U is weak there: U vs G -.037), N4 +.011 | +.013, N1 -.0075! | -.0058!, N2 -.0012! | -.0013!. Re-tuning b,c on tune seeds (pooled over the N worlds; b=.1) does not help (vs G held: N1 +.001/+.002, N2 -.015/-.006, N3 -.006/-.012, N4 -.007/-.006; declared / DX gate).
  => in 3 independent new generators + N4 the W4 bonus is a tie or a loss vs the best fair baseline; the only positive W4-style result remains the W4 generator family (C1-C3, M1-M2 and V3: +.05/.05/.03).
* V2 reproduced W4's failure (nbucb|learned -.0505/-.0453/-.0266 vs G held; W4 board: -.051/-.045/-.027). V4 (mine): nbucb|learned -.023/-.025/-.018, nbreg -.021/-.016/-.008, Holm, both sets. V3 (W4's mild variant): +.053/.050/.026 -- still wins.
* Gate fixes (vs plain UCB = own base without bonus, nbreg arm, held | confirm; W4 gain retained vs U in parentheses):
  V2: learned -.052/-.047/-.022 | -.052/-.048/-.023; RX(.1) -.019/-.017/+.001 | -.019/-.019/-.001; AB(.1) -.028/-.019/-.012; DXT -.0195/-.0167/-.0014 | -.018/-.015/-.004.
  V4: learned -.033/-.027/-.015 | -.035/-.030/-.015; RX -.011/-.009/-.003; DXT -.014/-.010/-.005 | -.014/-.012/-.006.
  W4 (gain vs U): learned +.077/.058/.024; RX +.045/+.033/+.016 (56-67% retained); DXT +.050/+.039/+.015 (64-67% retained).
  => RX/DXT cut the V2/V4 harm by ~60% and keep ~60-65% of the W4 gain, but every fixed arm still has a Holm-significant loss in V2 C1/C2 and V4 C1/C2 vs U (the remaining loss ~= the experiment cost). No gate tested "learns to turn the bonus off" cleanly. The correctness half of M1/M2 is unaffected.
  N4 ("unattainable" by routing) is NOT a harm world for the W4 bonus (+.003 ns): the harm needs the within-agent observational slope to be positive while switching agents buys nothing, which is what V2/V4 have and N4 does not. N4 is unattainable by ROUTING only: candidate selection still wins there (+.05, clairvoyant best-of-8 +.264).
Verdict I believe: W4's positive result should be stated as generator-specific; keep nbucb/nbreg lab-only. If any creative opt-in is ever exposed, best-of-k+dedup (cost-accounted, tag or learned-gated) is the more robust candidate but needs real graded outputs.

---
## WHERE IT HELPS / WHERE NOT (all synthetic; creative value vs best fair routing unless stated)
| situation | result |
|---|---|
| BoK-dedup k=3, W4-type worlds (6 creative cells) | +.033..+.067, both sets, Holm; cost +5 calls (break-even .01/call); diversity part +.016..+.036 over cost-matched best-of-k |
| BoK-dedup k=3, new generators N1,N3,N4 / N2 | wins everywhere (+.01..+.065) but diversity filter adds only in N3 (+.024), N4 (+.011..+.015); N1 ~0, N2 and NM2 -.004..-.013 (loss) |
| value misspecification V2/V4 (novelty unattainable by routing) | BoK-dedup still +.025..+.036; nbucb/nbreg -.02..-.05 |
| W4 nbucb/nbreg|learned (0 extra calls) | beats BoK-dedup k=3 in W4-type worlds (C1/C2/M) by .016..+.07; ties or loses in N1,N2,N3,N4; loses in V2/V4; best used for nothing outside W4 |
| gate fixes RX/AB/DX/DXT | harm cut ~60%, gain cut ~35-40%; V2/V4 loss remains Holm-significant |
| archive/MAP-Elites selection | does not tie dedup (loses W4/V2/V4/N1-N3); wins only N4 with designer-given cells |
| VS (assumed profile) | +.04..+.37 at kappa<=.4; flips to loss at kappa ~.5-.8; gate must be learned, tags unsafe (-.01..-.04 on correctness when wrong) |
| correctness worlds, correct tags | identical by construction; mixed-domain shared router: -.004..-.009 on the correctness half (borderline non-inferiority) |

## HONEST LIMITS
* All worlds mine/W4's; selectors and W4 value share the "distance to recent outputs" structure; N1/N2 have small headroom (clairvoyant best-of-8: +.057/+.044) so a modest win there is the ceiling.
* The judge/verifier sees realized quality (noise sd .05): real verifiers are noisier/biased; judge-free arms show what is lost. No real embeddings; outputs are 2-5 dim synthetic vectors; 8 agents; one task-stream design; single noise level.
* VS profile is an assumption; calls are priced equal, real prices differ (tokens vs calls). Only one k per method in the sensitivity tables beyond k=2,3,4 for BoK.
* Tuning of selector params is pooled over the world set (K/NK worlds included, where tags make them irrelevant); theta/w differ between W4-type and new generators, so a deployer must tune on their own data.
* held-out 31-60 was read before the final batch (see disclosure); confirm 61-90 agrees with held within .005 in every cell I compared.

## RECOMMENDATION
Not a default, not a merge candidate. The W4 gated novelty-bonus bandit does not generalise beyond its own generator family and has no cheap gate that removes its failure mode; keep it lab-only and label its result generator-specific. A tag/learned-gated best-of-k + dedup selector is the more robust creative opt-in in simulation but costs 2-3x calls, ~40% of its gain is plain best-of-k, and its diversity part fails on corpus-style novelty; it needs real graded outputs before any exposure. VS is a hypothesis about LLM output distributions, not a result.
