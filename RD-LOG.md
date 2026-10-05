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

Gate convention from Round 3 on: loss counts only if Holm-significant AND |diff| > tol (tol = .002 absolute, "noise level"); TOST(+-.01) reported. Baseline for rounds 3-16 = lib default hybrid alpha .7 (results/b1_baseline.json) unless stated.

## Round 3 — RRF(k=60) fusion instead of z-score fusion
Hypothesis: rank fusion is more robust than z-score. Result: domain B -.16..-.18 (r1/mrr), real_mrr CONF -.017, all Holm p<1e-5. Verdict: **REJECTED** (reproduces earlier h13 finding that RRF < z-fusion).

## Round 4 — corpus-mean centering of the dense channel
Result: no axis moves (|d|<=.015, nothing Holm-significant; real_mrr +.001). Verdict: **REJECTED (no gain)**.

## Round 5 — Rocchio pseudo-relevance feedback (beta .5, top-3) on hybrid
Result: domain B r1 -.061/-.051, mrr -.041/-.036 (Holm p<1e-4); real -.004 (n.s.). Verdict: **REJECTED** (small stores: top-3 contain wrong docs often, PRF drifts).

## Round 6 — soft domain-centroid prior (docs carry a domain tag/namespace; score += g*z(cos(q, centroid(domain(d)))))
Method: g=.5 and g=1.0, hybrid .7. Result g=.5: domain_r1 B +.039/+.061, mrr B +.035/+.050, route B +.065/+.096 (Holm p<=.013); no loss; real axis unaffected by construction (no tags). g=1.0 similar.
Verdict vs default: win. BUT see R7: dominated by R2, **not shipped** (no code added). Honest note: needs tagged docs; real-text equivalence is trivial (prior unused there).

## Round 7 — domain prior ON TOP of adaptive alpha (new baseline = R2 adaptive)
Result: g=.5: domain_mrr B CONF +.015 (Holm p=.033) but HELD +.002 n.s.; r1/route n.s.; g=1: nothing. No Holm gain in both splits. Verdict: **REJECTED** (adaptive alpha already captures what the prior gave).

## Round 8 — HARD domain routing (restrict candidates to top-1 / top-2 centroid domains) on top of adaptive
Result: top-1: domain r1 A -.093/-.099, B -.096/-.068 (Holm p<1e-4); top-2: A -.040/-.035 (p<3e-4). Verdict: **REJECTED loudly**: a hard domain classifier costs more recall (it errs on ~10% of queries) than the search-space shrink gains. Soft prior is the only safe form.

## Round 9 — int8 per-vector quantisation of doc+query vectors (bytes/vec 1536 -> 388)
Gate: all retrieval + semantic axes. Result: no Holm loss above tol (worst per-axis diff sem_classification -.0056 n.s.; real_mrr diff -.0001 Holm-significant but 1e-4). Already exists as `flatq` int8 tier in lib; this is a battery re-verification. Verdict: **KEPT (4x smaller, existing opt-in)**.
## Round 10 — float16 vectors (768 B): zero losses, TOST p=2.7e-48 for +-.01. Verdict: **KEPT (2x, existing opt-in)**.
## Round 11 — int4 per-vector (196 B): sentence_completion B CONF -.083 (Holm), analogies -.056. **REJECTED**.
## Round 12 — binary sign vectors (48 B): Holm losses on 11 tests (vocab -.20, analogies -.13..-.16, sentence_completion -.19..-.24, domain_mrr B -.04). **REJECTED loudly** (fine as a first-pass filter only, never as sole scorer).
## Round 13 — PCA to k dims (fit on the store's own vectors, f32): k=256/192 no Holm loss; k=128 real_mrr -.006 (Holm, > tol); k=64 real -.016/-.018 and domain -.01. k=192/256 KEPT only as a recipe component (see R14); k<=128 **REJECTED**.
## Round 14 — PCA + int8: PCA192+int8 = 196 B/vec (7.8x smaller than f32): no Holm loss above tol, real_mrr within TOST; PCA256+int8 260 B same; PCA128+int8 132 B: real_mrr -.006 REJECTED.
Verdict R14a: **KEPT as an offline recipe** (python `py/b1/eff.py::pca_q`). NOT ported to Rust (needs a projection matrix file + SVD fit; lib has int8/f16 but no projector). Caveat: PCA fit on store vectors; tiny stores (<~200 docs) have rank < 192 so "PCA192" is a full-rank rotation there.

## Round 15 — unsupervised whitening of the embedding space for semantic skills
Result (shrink 1e-4/1e-3/1e-2): massive losses (vocab -.11..-.17, classification -.26..-.08, analogies -.18); shrink .01 still vocab A -.11. Verdict: **REJECTED loudly**.
## Round 16 — prompt-templated word embeddings ("the meaning of {}", "the word {}") for analogies
TUNE: analogies .792 -> .831. Verify: analogies +.03..+.08 but Holm p>=.034 (HELD A p=.034 only) in a ~100-test family; no loss. Verdict: **REJECTED (not significant on both splits)**; underpowered axis (6 items/seed) — a larger analogy set would be needed to detect <=+.04.
