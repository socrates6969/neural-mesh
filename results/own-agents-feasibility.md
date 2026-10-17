# Can we build our OWN trainable AI agents? Feasibility study

Date: 2026-10-04. Read-only study; no installs, downloads, or code changes.
All figures are rough, representative orders of magnitude from general public knowledge, NOT measurements on this machine. No benchmarks were run.

Target hardware: AMD Ryzen 7 8700F, 16 GB RAM, NVIDIA RTX 5060 Ti (nvidia-smi on this box reports 8151 MiB, so assume **8 GB VRAM**, not 16), CPU-first, no datacenter.

---

## 0. Verdict up front

- **Today the intelligence is Claude: a frozen, ~hundreds-of-billions-class network we cannot retrain.** Our router/mesh/memory learn only to dispatch, team-assemble, and retrieve. No model weights change. That is not "our own neural network learning".
- **"Our own trainable agents" means owning and updating model weights.** That is possible, but only at small scale, and the result is much weaker than Claude.
- **Achievable now:** LoRA/QLoRA fine-tune a 3B-8B open-weight model locally (8 GB VRAM makes 7B-8B QLoRA tight but possible; 14B is not realistic). Hot-swappable adapters plus retrieval on top of our existing mesh.
- **Research-grade, not a product:** true continual / always-learning weights in an LLM without forgetting.
- **Infeasible here:** training anything competitive from scratch; any model that "self-improves into something smarter" on its own.
- **Recommended path:** keep Claude as the reasoner, treat a small local model as a cheap specialist, and keep investing in the router/memory layer where "gets better with use" is already real.

---

## 1. The key distinction

| Layer | What learns | Weights changed? | Status |
|---|---|---|---|
| Claude (reasoning, code, language) | nothing at runtime | No (frozen, not ours) | Fixed capability |
| Hebbian router / mesh / co-activation edges / health EMA | which agent/tool to dispatch, which teams work | Only our small routing tables | Built, works in simulation |
| Retrieval memory / SONA-style patterns | what to recall | Only stored records / small tables | Built |
| **Our own model weights** | the thing the user is asking about | Yes | Not built; this study |

Important caveat on current evidence: the router results in `results/` come from a **fixed simulation** (hidden success probabilities, trap task types, drift, node failures) with paired stats over seeds. That demonstrates the learning rule works as designed in that world. It is not proof of gains on real Claude workloads until validated on real task outcomes.

---

## 2. Options for owning trainable weights

### 2a. Train a small model from scratch

- **What fits locally:** roughly 10M-300M parameters. GPT-2-small class (~124M) is the practical ceiling for fun experiments on 8 GB; 1B from scratch is possible in principle but needs far more tokens and time than is sensible.
- **Compute reality:** a useful model needs on the order of ~20 tokens per parameter (Chinchilla-style rule of thumb) at minimum, and modern small models are trained on far more. A 124M model at ~2-3B tokens is days of GPU time on a consumer card; a 1B model at 20B+ tokens is weeks-to-months and still weak.
- **Data:** you need billions of clean tokens. Public corpora exist but downloading/curating is a large effort and each source needs license/safety auditing.
- **Quality (blunt):** toy-to-small. Coherent short text at best, little reasoning, no reliable tool use or instruction following unless you add a big post-training stage. It will not approach even a small modern open model, let alone Claude. Frontier models cost tens to hundreds of millions of dollars of compute.
- **Value:** educational, or a tiny single-purpose model (classifier, embedder, router head). Not an agent brain.

### 2b. Fine-tune an open-weight LLM locally (LoRA / QLoRA)

- **What the GPU handles (rough):**
  - 8 GB VRAM: QLoRA (4-bit base) on 3B-4B models comfortably; 7B-8B possible with short context, small batch, gradient checkpointing, and patience. 13B-14B QLoRA generally needs 16-24 GB, so not realistic here without CPU offload that makes it painfully slow.
  - Inference of a 4-bit 7B-8B model fits in 8 GB with modest context; larger context or 14B spills into 16 GB system RAM and slows sharply.
- **What it buys:** style, format, domain vocabulary, tool-call schema adherence, and narrow-task accuracy. It does not add general intelligence; it nudges a pre-trained model's behavior.
- **Data needed:** hundreds to a few thousand high-quality examples for a narrow behavior; more for broader changes. Quality dominates quantity. Our logged task/outcome records from the mesh could be a source, but outcomes would need cleaning and a privacy review.
- **Rough time:** a few hundred to a few thousand examples on a 3B-7B QLoRA is typically tens of minutes to a few hours per run on a consumer GPU. Iteration is feasible; hyperparameter and data-quality work is the real cost.
- **Risks:** overfitting small data, degrading general ability ("forgetting" the base model's skills), evaluation burden (need a held-out test set or you will fool yourself). Base model licenses must be checked before use.
- **Verdict:** the one genuinely practical route to "our own weights".

### 2c. Online / continual learning and plasticity ("evolving neural pathways")

Techniques people point to:
- **Hebbian / STDP / three-factor rules** (pre x post x reward-like modulator): what our router uses. Works well for small, shallow, low-dimensional mappings (feature -> agent). Does NOT scale into deep LLM representations as a drop-in training method; no accepted recipe trains a transformer LLM with local Hebbian rules to competitive quality.
- **Elastic Weight Consolidation (EWC) and similar regularizers:** penalize changes to weights important for old tasks. Helps on small benchmarks; weak and costly at LLM scale.
- **Experience replay / rehearsal:** mix old data into new training. Most reliable mitigation, but you must store and manage the data and the compute cost grows.
- **Adapters per task / model merging:** sidestep interference by not overwriting shared weights (see 2e).

Honest state of the art:
- **Catastrophic forgetting** is unsolved in general. Training on new data degrades old capabilities; the **stability-plasticity tradeoff** means more adaptivity costs more forgetting.
- **Continually learning LLMs are an open research problem**, not a product. Deployed systems get "better" by periodic offline retraining with curated data and evaluation gates, not by weights drifting continuously from live traffic.
- **Live weight updates from raw interaction are unsafe:** feedback noise, reward hacking, poisoning/prompt-injection feeding back into weights, and silent regression without a test gate.
- Therefore "constantly evolving neural pathways" inside an LLM is not something we can credibly promise. At the system level it is achievable (2e).

### 2d. Neuroevolution (NEAT, evolving topologies) and spiking nets

- **Neuroevolution (NEAT / HyperNEAT / evolution strategies):** works for small control policies, game agents, and small topology search, typically up to thousands-to-low-millions of parameters, where a fitness function is cheap to evaluate. Does not scale to training language models; gradient methods dominate there. Evolution strategies have been used as a gradient-free alternative at moderate scale in RL, but that is not our use case.
- **Spiking neural nets (SNN) / STDP:** niche. Real uses are neuromorphic hardware research, low-power event-based sensing, and computational neuroscience. On a conventional CPU/GPU they offer no accuracy or efficiency advantage and the tooling is immature for language. The repo already contains SNN experiments (`g8s_snn*`); those are research probes, not an LLM path.
- **Verdict:** interesting for small research toys or embedded controllers. Not a route to a capable agent brain.

### 2e. Adapter / memory-based "evolution without retraining"

This is the closest thing to "evolving" that works today.
- **LoRA hot-swap:** train small adapters (MBs, not GBs) per skill/project/user and load them on demand over a frozen base. Cheap, reversible, versionable; rollback is just deleting an adapter.
- **Retrieval memory:** store outcomes, notes, successful trajectories; retrieve them into context. Knowledge updates instantly with zero training and zero forgetting of the base model.
- **Routing / mesh (ours):** learn which agent/model/tool handles which task from outcomes; strengthen paths that work, decay and route around ones that fail.
- **Prompt / policy evolution:** evolve instructions, tool configs, and workflows against a test set (the `metaharness`-style loop). Gains are real but bounded by the underlying model.

Note this works with Claude too. It does not require owning weights. That is exactly what our mesh/memory already does at the system level.

---

## 3. Honest verdict and path

### Achievable now
- A local 3B-8B open model, QLoRA-tuned for narrow jobs (summarize logs, classify tasks, emit our tool-call format, draft low-risk text).
- Per-skill adapters that can be swapped and rolled back.
- Retrieval memory plus the Hebbian router choosing between Claude, the local model, and tools.

### Research-grade (do not promise)
- Continual learning of weights from live use without forgetting.
- Training from scratch to anything beyond toy quality.
- Hebbian/spiking replacements for transformer training.

### Infeasible on this machine
- Competitive from-scratch training; 14B+ fine-tuning at useful speed; frontier-class capability.

### Explicit statement
**Any model we train here will be far weaker than Claude on reasoning, coding, and broad knowledge, and it will NOT autonomously self-improve into something smarter.** Models do not bootstrap to higher intelligence by running; improvement needs better data, evaluation, and (usually) more compute, all supplied by humans and gated by tests. Self-training loops without strong external verification tend to degrade (drift, collapse, reward hacking).

### Credible step-by-step path (if pursuing owned weights)
1. **Define 1-2 narrow tasks** where a small model could plausibly be good enough (e.g., task-type classification, log triage, structured extraction). Write a held-out test set first, with success measured against Claude's outputs or real outcomes.
2. **Baseline without training:** prompt a small local model + retrieval; measure on the test set. Often this is already enough.
3. **Collect data** from mesh logs and Claude-generated/verified examples. Audit license, privacy, and quality; deduplicate; hold out test data. (Distilling Claude outputs may be subject to terms of service; verify before relying on it.)
4. **QLoRA fine-tune** a 3B-8B model on 8 GB VRAM. Compare to baseline on the held-out set. Keep only if it clearly wins.
5. **Package as a versioned adapter**, behind the existing router as just another candidate agent, with a health EMA so it is skipped when it fails.
6. **Periodic (not continuous) retraining:** batch new verified examples, retrain, evaluate, gate on regression tests, then promote. Keep replay data to limit forgetting.
7. **Only then** consider more exotic ideas (EWC, merging) as experiments, not commitments.

Cost realism: electricity and engineer time dominate; no cloud spend is required for steps 1-5. The hard part is evaluation and data quality, not GPU hours.

---

## 4. How this connects to what we built

- The router/mesh/memory already provide measurable "gets better with use" at the **dispatch and retrieval level**, with zero change to model weights: routing weights adapt to outcomes, co-activation edges learn good teams, health tracking routes around failures, memory recalls prior solutions.
- **Boundary:** this layer can pick the best available brain and feed it good context; it cannot make the brain smarter. A router over weak agents is bounded by those agents.
- **Evidence caveat:** the strongest numbers so far are from synthetic simulation with paired seeds. Next honest step is logging real routing decisions and outcomes and showing the same paired improvement on live traffic.
- **Fit for owned weights:** a local fine-tuned model would plug in as one more agent in the pool; the mesh would learn when to trust it. That makes the owned-weights path low-risk and incremental rather than a rewrite.

## Recommendation

Do not build "evolving neural agents" as a goal. Instead: (1) validate the mesh on real outcomes, (2) add a small local QLoRA specialist only where a measured test shows it beats a prompted baseline, (3) keep adapters and retrieval as the "evolution" mechanism with human-gated retraining, and (4) keep Claude as the reasoning engine.
