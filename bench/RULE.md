# Where does a memory go? (mesh store, local-LLM SFT, retrieval, ruflo mirror)

Evidence: bench/results/main (v1 front-end) and bench/results/v2 (identifier-aware front-end), 120-item battery, Claude haiku-4-5 with
MAX_THINKING_TOKENS=2048, local = Ollama qwen2.5-coder:7b (base, not fine-tuned), paired bootstrap 95% CIs, `python bench/analyze.py v2`.

| Memory kind | Goes to | Evidence |
|---|---|---|
| Volatile / repo-specific facts (constants, paths, config values, decisions, who-owns-what) | mesh store (credited) + retrieval at answer time. NOT SFT. | Private repo constants, n=15: Claude alone 0.00, local alone 0.13, Claude+retrieval 1.00, local+retrieval 1.00 (paired B-A +1.00 [1.00,1.00]). Weights go stale the day a constant changes; retrieval does not. |
| Stable skills and conventions (answer format, tool-call shapes, house style, how a role reports) | SFT rows, only if judge_score >= 0.7 and the promotion gate passes | Not measured here (needs the nightly adapter). The base 7B local model already completes 100% of format-constrained prompts but parses 94% of answers. |
| Exact computation (math, physics, unit work) | deterministic tools (neural-mesh tools.rs), not memory and not SFT | Local 7B: math 0.50, physics 0.17 vs Claude 0.87 / 1.00. Memory cannot fix arithmetic. |
| Routing outcomes (which agent did what, pass/fail, fix rounds) | router state via feedback + ruflo trajectories; summaries mirrored to ruflo "vector-patterns" | 69 real dispatches: label-fed mesh 67%, reward-only 33%; ruflo `hooks route` 19.0% -> 14.3% on 21 held-out rows after post-task/model-outcome/--train (no gain). |
| General world knowledge | neither: leave to Claude | Public stdlib facts: Claude 0.73, local 0.60, Claude+retrieval 0.80 (all CIs include 0). |

Rule of thumb: if the answer can change without the model changing, retrieve it; if it is a behaviour that should hold regardless, train it;
if it is computable, compute it. Every row written to SFT must also exist in the store (the store is the source of truth and the replay set).

Caveats: private-fact items were constructed so that the answer lives in the retrieval corpus (a best case for retrieval); the v2 front-end
(backticked identifiers and src/ paths -> restricted BM25) was written AFTER v1 retrieval scored 0/15 on them, so v2 is not a blind result.
Outside that category retrieval changed accuracy by -0.009 [-0.048,+0.019] (n=105).
