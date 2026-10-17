# G4 world definition freeze (pre-registration)

Frozen: 2026-10-04T06:37:30Z, BEFORE any diversity policy was written or run.
File: src/bin/g4_common/world.rs  sha256 = ea287f5903554394f24bd3e47a2061cac7b84d964d4363c843d2188ecfb36224
(The runner re-hashes the file at start and prints it; any mismatch means the world changed after freezing.)

Value definition (fixed): creative domain value = q * (0.25 + 0.75*nov); correctness domain value = q.
nov = min(1, nearest-neighbour distance of the new output embedding to the last 8 outputs of that domain, over the value dims).
Router reward = value (scalar). Fair baselines see only this scalar. Novelty-aware arms additionally see the output embedding
(obs dims) and must build their OWN novelty estimate (different window 10, k=3-nearest-mean, own scale); they never receive
Q, true styles or the value formula. Exceptions, labelled ORACLE-ISH: (a) declared gate = told which domains are creative,
(b) oracle-scorer arm = knows true styles + exact formula (ceiling only, not a candidate).

Worlds (9): C1 multistyle, C2 star+diverse, C3 clustered styles w/ drift & style-correlated quality  [creative];
K1/K2/K3 = same structures with value=quality only [controls, repeats fine];
M1 (3/6 creative), M2 (1/6 creative, star) [mixed];
R1 rigged check: value novelty is on hidden dims 2-4 but policies observe only dims 0-1 (observation uninformative).

Pre-registered pass bar for "creativity routing accepted": (1) beats best fair baseline (tuned per world on seeds 1-8) with Holm-
significant paired 95% CI > 0 on creative-domain value in >=2 of {C1,C2,C3} on held-out 31-60 AND again on fresh 61-90 with the
frozen config; (2) harmless: TOST equivalence (delta=0.01) vs best fair baseline on correctness-domain value in K1,K2,K3 and in the
correctness half of M1,M2, on both sets; (3) R1 shows no gain beyond what an id-only (non-observing) arm gets.
Diversity-arm hyper-parameters are tuned ONCE, pooled over all 9 worlds on seeds 1-8 (a deployer does not know which world they are in);
baselines are tuned per world on seeds 1-8 (generous to baselines).
