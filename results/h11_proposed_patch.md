# H11 proposed patches (R5-C token efficiency). NOTHING below is applied; I own only `src/bin/h11_*`, `results/h11_*`, `results/lessons_h11.md`.

All numbers referenced here are SYNTHETIC cost accounting (token-equivalents, TE) from seeded simulations, not real bills. See `results/h11_board_entry.md`.

## 1. `src/research.rs`: opt-in per-session RESOURCE counters (no pricing inside the library)
The loop already counts `calls`, `checks`, `flags`, `ctx_hits` in `SessionStats`. A cost model needs the SPLIT of those counts, because the study found that the
bill is dominated by generator completion + base prompt (48% + 19%), the verifier (21-41% when it is called on every step) and, only when documents are injected,
the injected context (up to 48%); encoder/retrieval compute is < 0.01% at API-style prices. Suggested additive fields (all `#[serde(default)]`, zero cost when unused):

```rust
// SessionStats: counts only; the caller multiplies by ITS prices
pub gen_calls: u64,          // draft + retry calls (today folded into `calls`)
pub verify_calls: u64,       // == checks today; keep both
pub ctx_items_injected: u64, // notes/docs actually put in a prompt (not merely retrieved)
pub retrievals: u64,         // searches paid (encode + scan), injected or not
pub memo_hits: u64,          // answers served without a generator call (see 3.)
// TaskReport: ctx_items: u32, verified: bool (already `checked`), served_from_memo: bool
```
A caller-supplied `fn tokens(&Context) -> usize` hook (default: chars/4) lets a deployment report injected-context tokens without the crate knowing a tokenizer.

## 2. Context size: default `ResearchConfig.k` (neighbours) is a ROUTING knob; add a separate `ctx_docs: usize` (default unchanged)
In the doc-QA study the cost-optimal number of injected docs was k = 1 for every doc-length scale and both exchange rates (real MiniLM-hybrid rank profile: E[#relevant in
top-k] = 0.94 / 1.20 / 1.30 / 1.38 / 1.44 for k = 1..5); k = 1 vs k = 3 cut the per-task cost by about one third at +0.012 success. The knee moves to 2-3 when the distractor
penalty eta <= 0.01 (analytic table in the board entry), so it MUST stay a tunable parameter with the default left as is. Suggested: `ctx_docs: Option<usize>` (None = today's behaviour).

## 3. Verified-answer memo (opt-in) with the three safeguards the controls required
`ResearchSession::run_task` could keep `HashMap<TaskKey, Entry{answer, concept, epoch, verified}>`:
* store only answers that a verifier passed (store-all WITHOUT eviction lost 0.01-0.50 success: -0.43 in the drift world, -0.50 in the weak-identification world, +90% cheaper but wrong);
* on a hit, re-verify with one verifier call before serving (catches ~75% of stale/wrong answers at the synthetic flag model's recall; verified-only without reverify still lost 0.26 in the weak-identification world and served 12.6% stale answers after drift);
* evict an entry when its delayed grade (`grade < succ_thr`) arrives (the existing `reuse` delayed-feedback path already delivers it; evict-on-bad-grade turned store-all from -0.15 to +0.05 mean success);
* do NOT enable it where the same text can carry different intents (the weak-identification world still lost -0.06 with all three safeguards);
* never key by embedding similarity alone: the semantic variant was strictly worse than exact keys in the off-topic control (wrong-served 9% vs 0.3%).
Gain depends only on the exact-repeat rate of the traffic (here 93% by construction in the main worlds, 3.5% in `ctl-uniq`: savings -3% there); measure the repeat rate on real traffic before enabling.

## 4. Verification gate in the session critic (g9/h3 gate family): break-even form
`SecondOpinion` decides from calibrated per-(feature, agent) estimates; the cheapest robust rule found here is `verify iff (1 - m) * Rv >= cv` with `m` = the router's own
posterior mean quality of (agent, step type), `cv` = verifier price relative to a generator step, `Rv` ~ 1.0 (tuned once on seeds 1-8, frozen). It cut cost 27% (treatment worlds)
at -0.0005 graded quality vs always-verify, and matched or slightly beat g18's online L4 on utility (S - best fair incumbent: +0.0007..+0.015 in 12 of 15 worlds; L4 still cheaper in lat-main/lat-highprop). A constant threshold `m < tau` failed where the verifier is expensive and where baseline quality is low
(first version kept in `results/h11_pipe_v1_constTau.*`). Only worth it when verification is more than ~15% of the bill; otherwise leave `critic` as is.

## 5. NOT recommended (measured, negative)
note/context compression to 50% or 25% of the tokens (cost -0.9..-2.4% only because extra retries eat the saving; quality -0.015..-0.08), score-gating retrieval beyond right-sizing k, learned per-anchor
context bandit, price-ordered cascade (quality -0.15..-0.26), best-of-3 without early stop (+136% cost), store-all/no-evict result cache, embedding-similarity result cache without reverify.
