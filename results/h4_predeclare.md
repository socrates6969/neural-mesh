# h4 pre-declaration (written BEFORE the held-out/confirm runs; tuning seeds 1-8 only)

Question: can the memory footprint we control (tiered memory / case store behind the router) be reduced under a hard byte
budget with routing quality PROVABLY held?  All numbers synthetic-until-validated-on-real-traffic.

Worlds (fixed before running, generators independent):
- A-zipf   : 60 recurring families, Zipf(1.1) popularity, 30% one-off noise tasks (useless cases accumulate), mid-stream drift of 25% of families (stale cases become misleading). K=6 agents, T=4000, eval window = last 40%.
- B-burst  : 300 bursty "lesson" families with time windows (stale after the window), shared-direction confusable geometry, log-normal per-family benefit (heavy-tailed value), 20% generic tasks, 0.6 junk trace entries/step. K=8, T=5000, eval = last 50%.
- Controls : A-allvalued (500 equally valuable families, no junk, no drift: forgetting MUST cost, report the bounded cost) and A-noreuse (nothing recurs: forgetting must be harmless).

Policies: full (reference), fifo, lru, value (recalled-and-helped utility, decayed), merge (value + importance-merge absorb), tiered-int8 / tiered-f16 (hot f32 + cold quantised; value eviction), plus nomem and ctxfree reference rows.
Budgets: fractions {50%,25%,10%} of the measured final bytes of the unbounded memory on the same seed (tiered also at 100%). Hard budget: enforced on every add (short ring of 64 items is exempt; any run whose peak exceeds the budget is counted in the `ovr` column).

Quality metric (primary): routing success on the eval window; secondary: useful-retrieval rate on recurring tasks.
Equivalence claim: paired (by seed, common random numbers) TOST vs full memory, margin +-0.01 success (strict tier +-0.005, both pre-declared),
Holm-adjusted over the main-world family {A-zipf, B-burst} x {fifo,lru,value,merge,tiered-int8,tiered-f16} x budgets.
"Memory reduced without quality loss" is claimed for a (world, policy, budget) cell ONLY if EQ holds in BOTH held-out (seeds 31-60) and confirm (61-90); otherwise the bytes-vs-quality frontier/knee is reported.
Frozen hyperparameters chosen on seeds 1-8 only (results/h4_tune.md); no post-hoc changes after held-out.

Addendum (written after tuning on seeds 1-8, BEFORE held-out/confirm): the tuning run showed value forgetting can IMPROVE on unbounded memory (stale/junk cases hurt), which TOST at +-margin
reports as "not equivalent" when the gain exceeds the margin. So a one-sided NON-INFERIORITY test is also pre-declared on the same family/margin (NI: H0 delta <= -0.01, Holm over the same family);
claims are stated as: EQ (TOST), NI (no loss beyond margin; includes gains), or neither. Frozen config after tuning: value hl=500 init=0.6; merge sim=0.8 (tuned with the pre-freeze value hp, disclosed); tiered-int8 hot=0.15 of budget; tiered-int8+merge hot=0.6; cold=int8/f16; short ring 64.
No world definition, policy or margin changes after this point.
