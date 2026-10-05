# RD-LOG — neural-mesh R&D director program (Rounds 0..N)

## Scope statement (read first, applies to every round)
The mesh/router are a **dispatcher + librarian in front of Claude**. This program improves routing, retrieval, memory, efficiency
and research assistance. It does NOT improve Claude's underlying IQ and is NOT "universal knowledge". Every battery item is a
**synthetic proxy authored by the R&D director** (templated / hand-written); scores are proxy scores for the mesh's own deterministic
components (embeddings, retrieval, routing, local tools), never IQ-test scores of Claude. Saturated axes (score ~1.0) are
**no-regress guards only**, not evidence of capability.

## Iron rules in force
Tune seeds 1-30 / held-out 31-60 / confirm 61-90; Holm over all (axis x world x split) tests of a candidate; >1 world (A=keyword-ish
phrasing, B=paraphrase+typos / chatty phrasing, R=real documentation text); TOST (+-0.01) for equivalence claims; KEEP only if >=1
target axis has a Holm-significant gain in the same world on BOTH HELD and CONF and NO axis has a Holm-significant loss; defaults
unchanged, new capability opt-in; commits local only, never pushed.

## Battery (py/b1/*, results/b1_*)  — the gate
* `py/b1/gen.py` generates 25 sub-skill proxies x 2 worlds x 90 seeds (6 items/skill/seed) + knowledge stores; `embed.py` = MiniLM (local ONNX).
* L1 routing (NL item -> capability: calc/units/csp/sandbox/semantic/knowledge): `route_kw` (hand rules), `route_knn` (MiniLM kNN trained on seeds 1-30).
* L2 exec/verify via the REAL Rust registry + Verifier (`src/bin/b1_exec.rs`): `exec_<skill>` (tool reproduces reference answer), `claim_<skill>` (accepts true / rejects plausible-wrong claim), `us_<skill>` latency. 19,440 items, 0 mismatches. Saturated (=1.0): regression guards only.
* L3 semantic (embedding-only arm): vocab, similarities, classification, verbal_analogies, sentence_completion.
* L3 knowledge (hybrid dense+BM25, lib default alpha .7): general_info, comprehension (sentence retrieval in passage), domain_r1/mrr/route (144 hand-written docs, 18 domains; STEM/math/physics/chem/bio/neuro/CS/biochem/computational-science, finance/quant-fin/business/marketing/digital-marketing/politics/corporate-mgmt/consulting/stock-market). Domain world A = keyword-sharing questions, B = paraphrase + 12% word-transposition typos.
* R: real documentation retrieval (h10 corpus A, 6,831 real paragraphs, real queries; seeds present: 1-8, 31-60, 61-90) `real_mrr`, `real_h1`.
* Regression: `cargo test --release --lib` = 333 passed (baseline).
* Skills with a clean automatic metric: digit span fwd/back, letter-number seq, mental arithmetic, numerical ops, quantitative problems, symbol coding/search, cancellation, series, matrix, inductive, deductive (CSP), rotation, paper folding, block design, visual puzzles — **all exec/verify proxies, saturated**. Proxy-only (explicit): pattern recognition etc. are generated from my own rule families, so "tool solves it" is in-distribution by construction; vocab/sentence completion are embedding proxies of verbal skill.

## Round 0 — Baseline (no change)
Method: `python py/b1/battery.py` (arms dense / bm25 / hybrid(.7)). Means over worlds A,B (seeds 31-60 | 61-90):
domain_r1 hybrid .775|.775, dense .844|.858, bm25 .605|.592; domain_mrr hybrid .852|.852, dense .898|.905; domain_route hybrid .878|.863, dense .910|.925;
real_mrr hybrid .363|(see json) vs dense .334 (hybrid wins on real text); vocab .667|.694; sentence_completion .647|.672; analogies .792|.822; classification .936|.936;
saturated: comprehension 1.0, general_info ~.98, similarities ~.99, route_knn .997, route_kw .96, all exec_/claim_ = 1.0.
Self-found flaw fixed during setup: first version injected query typos only into the BM25 channel (dense saw the clean text), which inflated dense; the typo'd text is now embedded too. Baseline above is the corrected one.
Verdict: baseline. Headroom axes: domain_*, vocab, sentence_completion, analogies. Files: results/b1_baseline.json.

## Round 1 — global alpha .9 instead of .7
Hypothesis: library default alpha .7 is too lexical for paraphrase-heavy domain queries. Method: alpha sweep on TUNE (0..1) -> best .9 (domain_mrr .8965 vs .8543); verify on HELD/CONF.
Result: world B domain_r1 +.126 (HELD) / +.150 (CONF), domain_mrr +.088/+.101, domain_route +.076/+.126, all Holm p<1e-3; world A ~0. BUT real_mrr CONF -.0098 (Holm p=.045) => regression on the real-document axis.
Verdict: **REJECTED as a universal setting** (workload-dependent knob: lexical corpora want .7, paraphrase corpora want .9+). No code change (HybridConfig already exposes alpha).

## Round 2 — query-adaptive alpha (`AdaptiveAlpha`, opt-in)
Hypothesis: choose alpha per query from lexical coverage of the best BM25 doc (strong match -> trust BM25, weak/paraphrase -> trust dense) so both regimes win.
Method: alpha_q = hi-(hi-lo)*cov^pow, cov = max BM25 / ((k1+1)*sum idf of in-vocab query tokens). Grid lo{.3,.5,.7} x hi{.9,1} x pow{1,2} on TUNE (domain_mrr+real_mrr) -> (.3,1.0,2.0). Frozen, then HELD/CONF.
Result: domain_r1 B +.142 (HELD) / +.154 (CONF), domain_mrr B +.096/+.103, domain_route B +.082/+.126 (Holm p<1e-5, wins/losses 28/1, 27/0); world A ~+.01 n.s. NO Holm loss on any of ~100 tests; real_mrr d=+.0027/-.0031 and is TOST-equivalent within +-.01 on both splits (p=.004/.006); general_info/comprehension unchanged.
Implementation: `hybrid::{AdaptiveAlpha, Bm25::lexical_coverage, HybridIndex::{fused_scores_adaptive, search_adaptive}}` (additive, defaults untouched). Parity test `tests/b1_adaptive_parity.rs` (Rust == python reference on 40 queries, scores within 5e-3, identical top-1). lib tests 333 pass.
Caveat: domain docs/queries are my own synthetic text (single-sentence docs); real-text equivalence is the only external evidence; B's typos are synthetic transpositions.
Verdict: **KEPT (opt-in)**.
