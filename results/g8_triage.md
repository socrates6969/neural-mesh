# G8 plugin triage (W8, read-only) - what in ruvnet-clean / ruvnet-crates could help neural-mesh?

Scope: read-only skim of C:\Users\mariu\ruvnet-clean (source repos) and C:\Users\mariu\ruvnet-crates (480 crates) against what neural-mesh
already has (DESIGN.md, results/research_board.md, src/*.rs, results/g*). Nothing was built, installed or run. Claims below are from reading
source/doc-comments, NOT from running anything; vendor performance claims are unverified.

## Bottom line
- Nothing found that is a strong, un-covered mechanism. Most of the retrieval/ANN/quantisation/oscillation/transfer/forgetting/hybrid-BM25 ground
  is already covered by our own experiments (d6, f1, f2, f4, f5-f8, g1, g2, g3, r1, tiered.rs, compact.rs).
- 5 small, honest experiment candidates (ranked at the end). Expect ties for 3 of them; none is a likely big win.
- Security: no miners/obfuscation seen in what I opened; a few hygiene notes (bottom).

## Per-candidate notes

### ruvector-sona (0.2.1) + ruflo SONA
- API: Micro/Base-LoRA, EWC++, ReasoningBank = k-means++ over trajectory embeddings -> `LearnedPattern`s with quality, `find_similar`, prune/consolidate; DarwinGuard, auto_tuner.
- Maps to: router (pattern memory) / knowledge anchors.
- Verdict: COVERED / irrelevant. Our sona.rs (ruflo port) and SonaPlus already tie guarded(hebr) and lose to keyword. LoRA/EWC need a NN; the Hebbian
  rows already have decay + CUSUM reset (hebbian `meta`). ReasoningBank k-means patterns ~= knowledge.rs leader-clustered anchors + casebased kernel-Beta.

### ruvector-tiny-dancer-core (FastGRNN router, VoI, circuit breaker)
- API: feature engineering (cosine/recency/frequency/success), FastGRNN scorer, circuit breaker, `voi::decide` (closed-form Gaussian value-of-information:
  buy a better estimator only if gain*value > cost; `s^2 = sigma^4/(sigma^2+tau^2)`, EI = a*Phi(z)+mu*(1-Phi(z))+s*phi(z)).
- FLAG (honesty): README markets "conformal prediction", "70-85% cost reduction". `uncertainty.rs` is a fixed `1-2|p-0.5|` heuristic and `calibrate()` is a
  literal `TODO`. No conformal anywhere in the crate. Treat the numbers as marketing.
- Circuit breaker == our mesh health EMA + bench (covered). FastGRNN needs trained weights + labelled features: no analogue in the sim (covered by casebased).
- VoI gate is the only novel piece: see experiment E3.

### ruvector-memory-admission (global-min-cut write-time cluster admission) + ruvector-namespace-merge
- API: streaming "merge into nearest cluster or open a new one" with 3 policies; the useful one is `AdaptiveMincutAdmission`: threshold = running Welford
  mean - k*std of the policy's own score distribution, bootstrap tau until 10 observations, no labels. Min-cut part is O(c^2) overkill for us.
- Maps to: knowledge.rs anchors (`sim_min`, hard-coded 0.6, and the board notes it "must exceed random-pair cosine tail ~3.5/sqrt(dim)" and needs retuning per
  embedder: hash vs MiniLM vs BGE).
- Verdict: EXPERIMENT-WORTHY (small). See E1.

### ruvector-agent-memory (coherence compaction, CAMA arbitration, mincut-gated forgetting, TARL ledger, witnessed compaction)
- Compaction: recency+frequency+context-coherence score; our tiered.rs already forgets by time-decayed outcome value (and has FIFO ablation). Coherence-to-recent-queries term is the only difference; no sim world where recent-context alignment predicts usefulness. COVERED / low.
- `graph_forget` (keep "bridge" memories so the store stays connected): needs multi-hop memory use; our retrieval is single-hop top-k. IRRELEVANT.
- `arbitration` (N memories declaring the same source count as ONE observation): real idea (memory correlation bias) but our sim has independent outcomes; would need a
  world with relayed/duplicated notes (research.rs notes?). See E5, low.
- Ledger / witness / signed-receipt crates (retrieval-receipt, proof-gate, rvf-*): provenance/tamper-evidence, unkeyed FNV-1a hashes ("keyless", their own words). IRRELEVANT to routing quality.

### ruvector-hybrid (BM25 + dense + RRF), agentic-search (BM25), ruvector-maxsim (ColBERT), ruvector-matryoshka, ruvector-gnn-rerank, ruvector-bounded-rag, ruvector-cluster-rag, ruvector-speculative-ann, ruvector-adaptive-ann, ruvector-entropy-ann, ruvector-recall-bounded, ruvector-query-cache, ruvector-diskann/acorn/spann/lsm-ann/rabitq/turboquant/pq-search
- Retrieval stack. g1_retrieval already measured: BM25 vs hash vs MiniLM vs BGE-base, hybrid lexical fusion (hash+hybrid, mini+hybrid, bge+hybrid), sentence-chunk
  max-pool (== maxsim idea), spell/keyword rewrite, Rocchio PRF, token-overlap rerank, cross-model cascade; mini+hybrid hit@1 ~.93, bge ~.94 (ceiling ~.95).
  d/f-series: MMR was a Holm LOSS in 9/9 worlds (so bounded-rag/diverse-beam diversity is the wrong objective). tiered.rs already uses turbovec (cold tier, exact
  rerank for hot set) = turboquant/rabitq/speculative-ANN pattern. Flat is fastest at our N (30k: 340us flat vs 1.3-2.5x HNSW at 11x memory).
- gnn-rerank (1-hop score diffusion over the candidate k-NN subgraph): ~= PRF; remaining headroom ~2 pts hit@1. COVERED-ish, not worth a slot.
- Matryoshka: MiniLM/BGE-base are not Matryoshka-trained; truncating dims would just hurt. IRRELEVANT.
- adaptive/entropy/recall-bounded ANN (auto-pick ef): irrelevant at flat-scan scale.
- query-cache: semantic cache of results; our f2_sched cache-gate already memoises best-known agent. IRRELEVANT (latency only).
- Verdicts: all COVERED or IRRELEVANT.

### ruvector-domain-expansion (Meta Thompson Sampling, DecayingBeta, transfer priors, PlateauDetector, CuriosityBonus, Pareto/hypervolume)
- Dampened-prior transfer across domains == f1 A1 (xfer-guard-map: real but conditional, loses when unrelated). DecayingBeta == ThompsonRouter gamma / discounted
  eps-greedy / hebbian decay. Pareto hypervolume == f1 C diagnostic. PlateauDetector (stop exploring when learning velocity ~0): we have fixed eps; CUSUM `meta` covers drift but not annealing; small, see below. COVERED. Curiosity bonus ~ UCB / novelty bonus (r2 rejected).
- Source of the one reusable idea for E2 (exponentially discounted per-bucket Beta).

### conformal-prediction (v2.0.0, uses lean-agentic + random-world)
- Real code: split/streaming conformal: exponentially-weighted conformal (EWCP, w=exp(-lambda*dt)) with a PID loop that adapts lambda to hold target coverage under drift; conformal predictive distributions.
- Maps to: Guarded hand-over gate (keyword vs learned) / abstain. d4 already tried abstain/calibration and got negatives; Guarded uses discounted-count Beta LCB.
- Verdict: low-value experiment E4 (calibrated coverage instead of Beta LCB). Honest expectation: tie.

### lean-agentic / leanr-*
- It is a Lean-4-style dependent-type kernel with hash-consing ("150x faster equality"). Nothing to do with agents or routing. IRRELEVANT.

### fact-tools (FACT)
- Not retrieval. A template + LRU cache + JSON "process" engine ("cognitive templates"). IRRELEVANT.

### SAFLA (Python)
- Core: 4-tier memory (vector/episodic/semantic/working, capacity eviction + consolidation), `DeltaEvaluator` = hand-weighted sum of 4 "deltas" (performance, efficiency, stability, capability) with context-adaptive weights, RL optimizer, safety boundaries, MCP server. No learned routing signal beyond what we have.
- COVERED (tiered/workmem/spatial/lessons) / IRRELEVANT. Heavy dependency surface (FastAPI, torch-ish, auth, deploy scripts).

### daa-* / ruv-swarm-* / synaptic-* / micro_swarm / rvagent-*
- daa-orchestrator 0.2.0 is a stub ("for the complete implementation see github"); daa-rules/economy are policy/market frameworks; ruv-swarm-daa "cognitive patterns" = r2 (REJECTED). ruv-swarm-core topology/distribution strategies = orchestration plumbing; ruv-swarm-ml = per-agent time-series forecasting (our drift handling is CUSUM). rvagent-subagents = LangGraph-style subagent spawning. IRRELEVANT/COVERED.

### ruvector-nervous-system (dendrites, HDC, Hopfield, dentate-gyrus separation, predictive coding, Kuramoto coherence routing, circadian gating, global workspace)
- Kuramoto/gamma/circadian == our r1_oscillations (REJECTED, controls matched). Predictive-coding residual gating == abstain/guard. Global workspace competition == mesh top-k activation. HDC/Hopfield/pattern-separation overlap with spatial.rs/r3 and casebased. COVERED / IRRELEVANT. (Crate also self-reports "5-50x savings", unverified.)

### ruvector-temporal-coherence
- cosine x exponential time decay x graph in-degree gate for retrieval. The recency factor is what E2 borrows; the in-degree gate is irrelevant (needs synthetic cluster topology, their corpus is random vectors).

### agenticow (COW branching vector memory), metaharness (GEPA/evolve harness minting), dspy.ts, SynthLang, local-logic, autogenous, guardrail, worldgraph, helix, RuView etc.
- COW branching: brains are KB-sized, clone is already free (fork_blank, f1). Harness/prompt evolvers and DSPy optimisers tune prompts, not agent-routing. Everything else is domain apps. IRRELEVANT.

### Other crates glanced at (names only): rvf-* (container format), rvm-*/ruvix-* (kernel/OS), qudag/daa-chain (crypto/DAG), ruqu-* (quantum), helix-*, wifi-densepose-*, nt-*/neuro-divergent (trading/forecasting), temporal-compare (DTW/LCS), midstream. No routing/memory content relevant to us.

## Experiment candidates (ranked)
| # | From | Idea (native reimplementation, std only) | Component | Metric / acceptance | Expectation |
|---|------|------------------------------------------|-----------|---------------------|-------------|
| E1 | memory-admission `AdaptiveMincutAdmission` | Self-calibrating anchor threshold: sim_min = running mean - k*std (Welford, bootstrap tau for first N) of the top-1 neighbour similarity, replacing hard-coded 0.6 | knowledge.rs anchors | Hold ONE frozen k across hash / MiniLM / BGE fixtures (f5/f6/f8 embeddings): anchor F1 / retrieval-vote success within TOST margin of per-embedder tuned sim_min, and no spurious anchors in the unique-embedding control (ctl-unique-ctx). Win = removes per-embedder tuning, not raw accuracy. | Moderate; top-1 sim stats mix intra- and inter-cluster so may need a trimmed/robust estimator |
| E2 | domain-expansion DecayingBeta, EWCP, temporal-coherence | Recency-discount per-anchor/per-agent vote stats (and optionally casebased kernel weights) with half-life h; add a CONTEXT-skill drift to the d6 worlds (today only sim_skill drifts) | knowledge.rs vote, casebased.rs | d6 treatment worlds + ctx-drift world: mean expected success at H500/H2000 vs undiscounted vote and vs best fair baseline, paired 95% CI, Holm; control ctl-irrelevant-ctx must not get worse | Small; the existing vote self-check gate (suppress if realised mean < front - margin) probably already absorbs most of it |
| E3 | tiny-dancer `voi::decide` | Replace SecondOpinion/probe heuristic triggers with a Gaussian VoI test: check iff VoI(mu,sigma from Beta posterior, checker noise tau, alternative a) * V > cost | variants.rs SO/HierSO, mesh probes (d1/x2/d2 worlds) | net-of-cost success vs HierSecondOpinion, calls/task, none-world no-harm loss (-0.0005 today) | Low; closed form buys principled thresholding, not new information. Note d1 found router-uncertainty triggers do NOT work (failure-streak does) |
| E4 | conformal-prediction EWCP/PID | Drift-aware conformal lower bound for the Guarded hand-over instead of Beta LCB | prior.rs Guarded | x1 worlds, traps8/correct8/drift: success vs guarded(hebr) | Low; d4 calibration negatives suggest a tie |
| E5 | agent-memory `arbitration` | Count records with a shared declared source as one observation in votes/lessons | knowledge.rs vote, research.rs notes | New world with relayed duplicate notes (own construction, so partly self-fulfilling): success vs naive vote | Low; only matters if duplicates are real in deployment |

Do-not-spend list (already negative or covered here): hybrid BM25/RRF, MMR/diversity, PRF/rerank, Matryoshka/quantised ANN, oscillation/Kuramoto/circadian gating, SONA/LoRA/EWC, transfer-prior TS, cognitive-pattern diversity, query caches, memory compaction by coherence.

## Security / hygiene notes (passing, not an audit)
- No miners or obfuscated payloads seen in what I opened. Prior scan_report: 0 MINERS, 5 download failures (ruvector-delta-index, ruvector-mincut-gated-transformer, ruvector-sparsifier, rvcsi-adapter-nexmon, intrinsic-dim: HTTP 403/500, i.e. not scanned).
- build.rs with process spawn: ruvector-lsm-ann (runs `rustc` for version), cuda-rust-wasm (pkg-config / nvcc probes), daa-prime-core (checks for `protoc`). Benign by inspection (no network), but still arbitrary code at build time: do not `cargo build` them without review. 27 other plain build.rs exist.
- "Telemetry" string hits: cognitum-one (agentic/telemetry*.rs, internal metrics), aimds-core gate, code-mesh config; JS side: metaharness packages (OpenTelemetry attribute mapping, darwin-mode bench telemetry.mjs, mutators calling openrouter/requesty: network by design, API-key driven), local-logic/poker `phoenix_config.py` (LLM tracing config). I saw no phone-home-on-import, but I did not trace each one.
- `ruvnet-clean` is NOT purely source: install-all.log shows `npm install` ran for worldgraph; agentic-difusion contains a full Windows `.venv` (Lib/site-packages incl. fastapi/huggingface_hub telemetry modules); agentic-search (91 pkgs), obsidian-brain (105) and agenticow have `node_modules`. dspy.ts/models ships binary `.onnx` files (7 MB text-generation.onnx): do not load.
- Vendor claims ("150x", "83x", "113x", "70-85% cost reduction", "5-50x") appear throughout READMEs; none verified here, and at least one (tiny-dancer conformal) is contradicted by its own source.
