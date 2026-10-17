# Market research: learned LLM router + shared memory layer

Prepared 2026-10-04. Read-only desk research (web search/fetch only). Every figure is tagged with source and date.
Confidence tags: [PRIMARY] = read on the originating page; [SECONDARY] = reported by a third-party page/search summary, not seen on the primary; [ESTIMATE] = my own derivation or unverifiable.
Note: several primary pages (Gartner, OpenAI marketing page) returned 403 or were read through a summarizer, so those figures are [SECONDARY] and should be re-checked before they go on a slide.

Positioning used throughout: a learned router plus retrieval/shared-memory layer that sends each request to the cost-optimal model, fails over around bad models, and shares a vector memory across agents. It is not neuromorphic, not a new model, and not yet proven on production traffic.

---

## 1. Market size

### Observed spend (last ~3 years)
| Metric | Figure | Source / date | Tag |
|---|---|---|---|
| Total enterprise gen-AI spend (apps + infra) | $11.5B (2024) -> $37B (2025), 3.2x | Menlo Ventures, "2025 State of Generative AI in the Enterprise", 2025-12-09; survey of 495 US enterprise AI decision-makers, Nov 7-25 2025 | PRIMARY |
| Foundation-model API spend (closest proxy to "tokens bought") | $12.5B in 2025 (largest infra category) | same Menlo report | PRIMARY |
| Enterprise LLM spend (earlier definition) | $3.5B (Nov 2024) -> $8.4B (mid-2025) | Menlo Ventures mid-year update, GlobeNewswire 2025-07-31; survey of 150 technical leaders | PRIMARY |
| Enterprise share of LLM API spend | Anthropic 40%, OpenAI 27%, Google 21%, others 12% (2025) | Menlo, Dec 2025 | PRIMARY |
| Provider switching | only 11% of teams changed provider in past year; 66% upgraded models within same vendor | Menlo mid-year, 2025-07-31 | PRIMARY |
| Worldwide AI spending (all categories, mostly infrastructure) | ~$2.6-2.7T in 2026, $3.64T in 2027; "Generative AI models" line $28.3B (2026), $51.6B (2027) | Gartner press releases May and Sept 2026, via search summaries (gartner.com fetch returned 403) | SECONDARY |

Caveats: Menlo is a venture firm surveying US enterprises, so it is a sample-based estimate, not billing data. Menlo's own two datapoints use different scopes. Gartner's "$2.7T" is dominated by data-center and hardware spend and is not the token market. The 2023 baseline is not in the sources I read, so I am not quoting one.

### Projections (next ~10 years)
Published forecasts are for the broader "generative AI market" (software + services + hardware, definitions vary), not API tokens. They disagree widely, which is itself a warning:
- ABI Research: gen-AI software $63B (2025) -> ~$220B (2030), ~29% CAGR. [SECONDARY, search summary]
- MarkNtel: $28.9B (2024) -> $142.7B (2030). [SECONDARY]
- Global Market Insights: $83.3B (2026) -> $988B (2035), 31.6% CAGR. [SECONDARY]
- Market.us: ~$1.08T by 2035, 35.6% CAGR. Emergen: ~$1.2T by 2035. InsightAce: $22.3B (2025) -> $678B (2035), 40.8% CAGR. [SECONDARY]
- Grand View Research, enterprise gen-AI only: $2.9B (2024) -> $19.8B (2030), 38.4% CAGR. [SECONDARY]
Spread at 2035 is roughly $0.68T to $1.2T. Treat these as low-reliability syndicated-report numbers. I would not cite any single one as "the" TAM.

### TAM / SAM / SOM framing (all [ESTIMATE], my arithmetic, not analyst figures)
- TAM (spend a router could in principle touch): third-party LLM API spend. Anchor: ~$12.5B in 2025 (Menlo), plausibly $25-50B by 2027 if you read Gartner's $28.3B/$51.6B "models" line as a rough cross-check. Different definitions, so a range, not a point.
- SAM (spend where routing is practical): workloads with a meaningful price gap between models and a share of easy requests, using a multi-provider setup. Unknown share. If you assume 20-40% of API spend is routable and addressable (pure assumption, needs validation), 2025 SAM is ~$2.5-5B of spend. What a vendor captures is a fee (OpenRouter ~5% take) or a share of savings, i.e. low single-digit % of that spend.
- Revenue sanity check from the nearest comparable: OpenRouter ARR ~$160M (Aug 2026, Sacra) at ~5% take implies roughly $3B of annualized spend flowing through it. [ESTIMATE derived from SECONDARY figures] This shows the routing/aggregation market already supports nine-figure revenue, but it is dominated by an incumbent.
- SOM: cannot be estimated honestly without pilot data. Do not present one.

Pain-point context (supports demand, but weak sources): Gartner (June 2025) predicts >40% of agentic AI projects will be canceled by end of 2027 citing escalating costs, unclear value, or inadequate risk controls [SECONDARY via hpcwire/bigdatawire summary]. Blog posts claim 73% of enterprises exceeded AI cost projections (attributed to FinOps Foundation 2026) [SECONDARY, unverified; I did not see the report]. Agentic workflows claimed to use 5-30x tokens per task [SECONDARY, blog, unverified].

---

## 2. Competitors (this is an active, crowded space)

| Player | What it does | Open / commercial | Status (sourced) | Overlap with us |
|---|---|---|---|---|
| OpenRouter | Single API to 400+ models/60+ providers, auto-routing, fallbacks, ~5-5.5% fee | Commercial (hosted) | $113M Series B May 2026 at ~$1.3B; Bloomberg reported Aug 16 2026 that Stripe agreed to acquire it for >$7B; ARR ~$160M Aug 2026 (Sacra). Seed a16z, Series A Menlo. [SECONDARY, news/Sacra] | Direct: routing + reliability. Huge distribution. |
| Martian | "Model mapping" learned router, 200+ models, claims 20-97% cost reduction | Commercial | $9M (NEA, Prosus, General Catalyst, Accenture Ventures), 2024. Blog reports it "nearing" $1.3B valuation Apr 2026 [SECONDARY, unverified]. Vendor figures, not independent. | Direct: learned routing. |
| Not Diamond | Learned "meta-model" router, prompt adaptation, agent-oriented routing; pay-go ~$0.05 per 1M tokens routed | Commercial | Backers incl. IBM, SAP.iO, Jeff Dean, Julien Chaumond; ~$2.3M pre-seed late 2025 per blog [SECONDARY] | Direct: learned routing + prompt adaptation. |
| RouteLLM (LMSYS/UC Berkeley) | Open-source framework for training/evaluating routers on preference data (ICLR 2025) | Open source | Reference academic baseline. [PRIMARY: lmsys.org blog 2024-07-01] | Direct technical prior art for the routing part. |
| Unify AI | Router/benchmark-driven model selection | Commercial | Was a RouteLLM comparison target; I did not verify its current status. | Direct, status unclear. |
| LiteLLM (BerriAI, YC) | Open-source gateway/proxy to 100+ providers, spend tracking, budgets, fallback, load balancing, semantic cache | Open source + enterprise | Widely used self-hosted. [SECONDARY] | Reliability, cost control; limited learned routing. |
| Portkey | AI gateway + observability, Apache-2.0 gateway | Open + commercial | $15M Series A Feb 2026; acquired by Palo Alto Networks, announced Apr 30 2026, closed May 29 2026, ~$140M. [SECONDARY] | Gateway/reliability/guardrails. |
| Requesty | LLM gateway/router, 300-600+ models, fallbacks, caching, EU/GDPR routing | Commercial | ~$3M raised 2024. [SECONDARY] | Direct for routing/fallback. |
| Cloudflare AI Gateway, Vercel AI Gateway | Managed gateways: caching (mostly exact-match/provider-native), rate limit, analytics, multi-model access | Commercial, platform-tied | [SECONDARY, comparison blogs] | Reliability/observability; distribution via platforms. |
| Other gateways (Bifrost, TrueFoundry, Kong, etc.) | Failover, semantic caching, MCP gateways, quotas | Mixed | [SECONDARY] | Reliability, cost governance. |
| Memory layers: Mem0 (~$24M raised), Letta (~$10M seed), Zep/Graphiti | Persistent/shared agent memory, scoped per user/agent/session; Letta has multi-agent shared memory blocks | Open source + commercial | Funding per a VC-blog roundup, mid-2026 [SECONDARY, unverified] | Overlaps the shared-memory half. |

Where a learned-routing + shared-memory layer could differ (hypotheses, not proven):
- Bundling routing with a fast shared vector memory across agents. Competitors mostly do one half (routers or memory), so a combined offering is a differentiation hypothesis. Whether customers want them bought together is untested.
- Reuse of retrieved context could cut prompt tokens independently of model choice. Unmeasured.
- Failing-model avoidance is table stakes (OpenRouter, LiteLLM, Portkey, Requesty all offer fallback). It is not a differentiator.

Honest assessment: routing and gateways are a consolidating market with well-funded winners (a reported $7B+ OpenRouter deal, a $140M Portkey exit). That validates demand, and it also means the incumbents have distribution, provider relationships, and large traffic datasets to train routers on. Our edge is unproven until validated on real traffic.

---

## 3. Realised savings from routing (heavy caveats)

| Claim | Number | Source | Independence |
|---|---|---|---|
| RouteLLM, MT-Bench | >85% cost reduction at 95% of GPT-4 quality (about 14% of queries sent to strong model) | LMSYS blog 2024-07-01; arXiv 2406.18665 [PRIMARY] | Authors' own; two-model setup (GPT-4 Turbo vs Mixtral 8x7B) |
| RouteLLM, MMLU / GSM8K | 45% / 35% cost reduction at 95% quality | same | Authors' own. Needed training-data augmentation; near-random on MMLU without it; generalisation tested on only one other model pair |
| Martian | "20% to 97%" cost reduction | blog summarising vendor claims [SECONDARY] | Vendor, unverified |
| Not Diamond | "20-40% savings" (secondary); one customer case "+10% quality, -10% cost and latency" (VentureBeat via blog) | [SECONDARY] | Vendor/customer anecdote |
| Generic blog ranges | "30-85%", "60-75% on mixed workloads" | assorted 2026 blogs [SECONDARY] | Marketing, no methodology |

Independent evidence is much less flattering:
- LLMRouterBench (arXiv 2601.07206, 400K+ instances, 21 datasets, 33 models) [PRIMARY]: "several recent routing methods, even including the commercial router OpenRouter, do not outperform a simple baseline (the Best Single model)". Best routers got up to ~4% accuracy gain or up to ~31.7% cost reduction at best-single-model accuracy. A large gap to an oracle remains. This is the most defensible "realistic" savings number I found: tens of percent, not 85-97%, in a neutral benchmark.
- RouterBench (arXiv 2403.12031, from Martian-affiliated authors): not all routers beat the best single model; top methods up to 31.7% cost reduction. [SECONDARY via search summary]
- RouterArena (arXiv 2510.00202, Rice University, 12 routers): no router optimal across metrics; trade-off between accuracy and cost. A blog claims a leading commercial router ranked 12th for over-selecting expensive models. [SECONDARY for the ranking claim; I could not read the leaderboard]
- Economics: a router adds its own classifier/embedding latency and cost, and only pays off with a large price gap between models and a meaningful share of easy traffic. [SECONDARY, blog]
- Results are benchmark-based. Real production traffic distributions, quality tolerance, and agentic multi-step tasks differ, and published studies mostly use single-turn QA.

Recommended pitch stance: say "published, benchmark-based savings range from roughly 30% to 85% depending on the benchmark and who measured it; neutral evaluations show the low end and sometimes no gain; we have not measured our own on production traffic." Do not quote 85-97% as expected.

---

## 4. Target segments and buying pain

Segments (hypotheses, ranked by my read of fit):
1. AI-native startups with a large and growing token bill and engineers who will self-serve (shortest sales cycle; also the segment OpenRouter/LiteLLM already serve).
2. Agent-platform builders running many agents with repetitive context (best fit for the shared-memory half and retry/fallback logic).
3. Enterprises with large LLM bills and multi-provider requirements (longer cycle; need security, data residency, SSO, audit; incumbents like Portkey/Palo Alto and cloud vendors are strong here).

Buying pain, with support strength:
- Token cost: agentic workflows multiply calls per task; blog claims of 5-30x tokens per task and bills rising despite falling unit prices [SECONDARY, unverified]. Gartner's >40% agentic cancellations citing costs [SECONDARY]. Menlo shows spend up 3.2x in a year [PRIMARY].
- Reliability: provider outages and rate limits motivate failover. Even OpenRouter had two outages of ~35-38 minutes on Feb 17 and 19 2026 per one blog [SECONDARY].
- Observability/governance: spend tracking, budgets, per-team quotas are standard gateway features.
- Counterpoint on stickiness: Menlo found only 11% of teams switched providers in a year. Buyers are often locked into one vendor and chase newer models from the same vendor, which limits multi-model routing upside. Per-token price drops (and vendor cheap tiers) also shrink the absolute savings pool over time.

---

## 5. Per-token list prices (USD per 1M tokens, standard tier)

Read 2026-10-04. Prices change often; recheck the day you ship the calculator.

| Provider / model | Input | Output | Source | Tag |
|---|---|---|---|---|
| Anthropic Claude Fable 5.1 | $10 | $50 | platform.claude.com/docs/en/about-claude/pricing | PRIMARY |
| Anthropic Claude Opus 5.5 | $4 | $20 | same | PRIMARY |
| Anthropic Claude Opus 5 / 4.8 / 4.7 / 4.6 / 4.5 | $5 | $25 | same | PRIMARY |
| Anthropic Claude Sonnet 5.5 and Sonnet 5 | $2 | $10 | same (Sonnet 5 $2/$10 is now standard; the planned Sept 1 2026 rise to $3/$15 was cancelled) | PRIMARY |
| Anthropic Claude Sonnet 4.6 / 4.5 | $3 | $15 | same | PRIMARY |
| Anthropic Claude Haiku 4.5 | $1 | $5 | same | PRIMARY |
| OpenAI gpt-6-astra | $10 | $50 | developers.openai.com/api/docs/pricing (via fetch summarizer) | SECONDARY |
| OpenAI gpt-6.1-sol | $2 | $10 | same | SECONDARY |
| OpenAI gpt-6-luna | $0.10 | $0.50 | same | SECONDARY |
| OpenAI gpt-5.4-mini | $0.75 | $4.50 | same | SECONDARY |
| OpenAI gpt-5-nano | $0.05 | $0.40 | same | SECONDARY |
| Google Gemini 3.5 Flash | $1.50 | $9.00 | ai.google.dev/gemini-api/docs/pricing (via fetch summarizer) | SECONDARY |
| Google Gemini 3.7 / 3.8 Flash | $0.75 | $3.75 (rises to $1.50/$7.50 on 2027-01-01) | same | SECONDARY |
| Google Gemini 3.5 Flash-Lite | $0.30 | $2.50 | same | SECONDARY |
| Google Gemini 2.5 Flash-Lite | $0.10 | $0.40 | same | SECONDARY |

Notes for the calculator:
- Aggregator pages conflicted with the official ones on OpenAI model names/prices (e.g. a "GPT-5.6 Sol $4/$20 promo"). I used the official OpenAI page as read, but it was summarized by a tool, so verify manually.
- Anthropic: batch API is 50% off; cache reads are 0.1x input (0.05x on Opus 5.5, 0.025x on Fable 5.1); Claude 4.7+ tokenizers produce ~30% more tokens for the same text, so "per-token" comparisons across vendors understate real cost differences. US-only inference is 1.1x.
- Baseline comparators already give big discounts without a router: batch (50%) and prompt caching (90% on cache reads). A fair savings calculator must compare against a caching/batching baseline, otherwise it overstates router savings.
- Price spread that makes routing worthwhile: premium (~$10/$50) vs cheap (~$0.10-$1 input) is 10-100x on input, which is the headroom. The headroom shrinks when the premium tier is itself cheap (e.g. Sonnet at $2/$10).
- Illustrative formula (not a measured result): savings% = share_routed_to_cheap x (1 - cheap_price/premium_price) - router_overhead - quality_loss_penalty. Example, for illustration only: 50% of tokens moved from a $10/$50 model to a $1/$5 model saves 0.5 x 0.9 = 45% before overhead. The 50% routable share is an assumption, not data.

---

## SWOT (sober)

Strengths
- Real, large, fast-growing underlying spend: $12.5B model-API spend in 2025 and 3.2x total enterprise gen-AI growth (Menlo).
- Large price spreads between model tiers (10-100x) leave real headroom.
- Combining routing with shared agent memory is a coherent hypothesis for agent-heavy workloads, and few competitors bundle both.
- Could position as open/low-cost entrant where incumbents charge ~5% take.

Weaknesses
- No production-traffic validation; savings and quality retention are unmeasured. This is the key weakness.
- No distribution, brand, provider relationships, or traffic data relative to OpenRouter, LiteLLM, Portkey (Palo Alto), cloud gateways.
- Router quality is hard: neutral benchmarks show many routers, including commercial ones, fail to beat the best single model.
- Not a model, not neuromorphic; no proprietary model moat. Technique (learned routing) has public prior art (RouteLLM, RouterBench, LLMRouterBench).

Opportunities
- Agentic workloads with many calls and repeated context may raise both routing and memory value.
- Neutral, auditable measurement could be a wedge: independent benchmarks show vendor claims are inflated, so a vendor that publishes honest, reproducible results on customer traffic stands out.
- Enterprises wanting self-hosted/EU-resident routing as hosted incumbents consolidate into larger platforms (Stripe-OpenRouter, Palo Alto-Portkey), possibly creating openings for independent options. [ESTIMATE/speculative]
- Failover and cost governance are bought as ops/FinOps tooling; savings can be shown in dollars quickly in a pilot.

Threats
- Consolidation: well-funded incumbents with captive traffic can add shared memory or better routing quickly; memory specialists (Mem0, Letta, Zep) can add routing.
- Model vendors cut prices and add cheap tiers, caching, and batch discounts, shrinking the savings pool; frontier vendors can also bundle auto-routing (e.g. tiering within their own families).
- Low switching: only 11% of enterprises changed provider in a year (Menlo), limiting multi-vendor routing demand.
- Quality-regression risk: one bad routing decision on a premium-quality task can destroy trust; enterprises may prefer "buy failover, not judgment".
- Benchmarks may not transfer to agentic production traffic; claims that cannot be reproduced will be discounted by sophisticated buyers.

---

## What to say, and not say, in the pitch
- Say: "Active, funded category; demand evidence is strong; published routing savings are about 30-85% on benchmarks, with neutral studies showing the lower end; we will validate ours on real traffic in a pilot and publish methodology."
- Do not say: expected savings of 85-97%; "neuromorphic"; "new model"; any SOM figure; that competition is weak.
- Highest-value next step to de-risk: replay a design partner's real request log through the router against a best-single-model and a caching/batch baseline, and report cost, quality (judged), and latency overhead.

## Source list
- Menlo Ventures, 2025 State of Generative AI in the Enterprise (2025-12-09): https://menlovc.com/perspective/2025-the-state-of-generative-ai-in-the-enterprise/
- Menlo mid-year update via GlobeNewswire (2025-07-31): https://www.globenewswire.com/news-release/2025/07/31/3125037/0/en/enterprise-llm-spend-reaches-8-4b-as-anthropic-overtakes-openai-according-to-new-menlo-ventures-report-on-llm-market.html
- Gartner AI spending forecast (Sept 2026; 403 on fetch, figures via search summary): https://www.gartner.com/en/newsroom/press-releases/2026-09-16-gartner-forecasts-worldwide-ai-spending-to-grow-49-point-5-percent-in-2026
- Gartner agentic cancellations (via summary): https://www.hpcwire.com/bigdatawire/this-just-in/gartner-predicts-over-40-of-agentic-ai-projects-will-be-canceled-by-end-of-2027/
- Syndicated gen-AI forecasts: https://www.gminsights.com/industry-analysis/generative-ai-market, https://www.abiresearch.com/blog/generative-ai-software-market-report-summary, https://www.grandviewresearch.com/horizon/outlook/enterprise-generative-ai-market-size/global
- RouteLLM: https://www.lmsys.org/blog/2024-07-01-routellm/ and https://arxiv.org/pdf/2406.18665
- LLMRouterBench: https://arxiv.org/html/2601.07206v1 ; RouterBench: https://arxiv.org/abs/2403.12031 ; RouterArena: https://arxiv.org/abs/2510.00202
- OpenRouter: https://sacra.com/c/openrouter/ ; Stripe deal report: https://finance.yahoo.com/technology/ai/articles/stripe-acquires-openrouter-7b-turning-091812340.html
- Portkey acquisition: https://www.respan.ai/blog/llm-observability-consolidation ; https://chatforest.com/reviews/portkey-ai-gateway-review/
- Martian: https://pulse2.com/martian-llm-router-company-raises-funding-from-accenture-ventures/ ; router landscape blogs: https://www.bestaiweb.ai/openrouter-martian-and-not-diamond-the-2026-llm-router-race-and-where-agent-cost-optimization-is-heading/ , https://dreaming.press/posts/2026-06-21-routellm-vs-notdiamond-vs-martian.html
- Not Diamond: https://www.notdiamond.ai/pricing ; Requesty: https://www.requesty.ai/ ; LiteLLM: https://github.com/BerriAI/litellm
- Gateways comparison: https://api7.ai/vercel-vs-cloudflare-ai-gateway ; https://dev.to/sebuzdugan/top-5-enterprise-ai-gateways-2026-bifrost-vs-kong-vs-litellm-vs-cloudflare-vs-vercel-3i10
- Memory layers roundup: https://mnemoverse.com/docs/library/ai-memory-solutions-2026-q3 ; https://valueaddvc.com/blog/the-ai-memory-problem-how-startups-are-solving-for-persistent-context
- Prices: https://platform.claude.com/docs/en/about-claude/pricing ; https://developers.openai.com/api/docs/pricing ; https://ai.google.dev/gemini-api/docs/pricing
