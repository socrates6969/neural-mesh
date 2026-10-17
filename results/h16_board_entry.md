# H16 (R8-A) board entry: productionised opt-in research-pipeline pieces (library types + tests + parity)

Owner: worker R8-A. Files: `src/hybrid.rs`, `src/verifygate.rs`, `src/lib.rs` (+2 lines `pub mod hybrid; pub mod verifygate;`), bins `src/bin/h16_hybrid.rs`, `src/bin/h16_gate.rs`, results `results/h16_*`, `results/lessons_h16.md`.
No other src file, no Cargo.toml change, no downloads, defaults unchanged (nothing is wired into research.rs / knowledge.rs / serve).
This is a PRODUCTIONISATION of already-verified (h13, g9, h3, h11) results: no new science claim. All performance numbers are on seeded simulations or the real-text h10 fixtures (synthetic-until-validated-on-real-traffic label applies; the end-to-end routing effect of the hybrid retriever is a TIE per h13).

## What was built
* `neural_mesh::hybrid`: `tokens`, `zscore`, `top_n`, `Bm25` (k1 1.2, b .75), `HybridIndex` (dense dot + BM25, z-score fusion, `alpha` default .7, `search / fused_scores / dense_scores`, serde), `SpellCorrector` (document-vocabulary OSA-1 corrector, serde, `with_known`).
* `neural_mesh::verifygate`: `FixedGate` (+ `fit_break_even` = h3 R1, `fit_capture` = h3 R2), `calibrate_gate` (AUROC >= .6 and >= 30 wrong guard, falls back to always-verify, returns a `CalibrationReport` with AUROC / ECE raw / ECE recalibrated), `ShiftMonitor` (error-rate band re-calibration trigger), `PosteriorGate` (h11 `verify iff (1-m)*rv >= cv`, nmin), the g9 evidence verifier core (`claim_verdict`, `task_verdict`, `EvidenceRule {SupportAny, SupportEntity, Latest, Agree}`, `Verdict`), `PrecisionGate` (online flag-precision gate with delayed feedback queues) and `EvidenceCritic` (assess -> decide {Accept, Reroute, DeliverUnverified} -> after_reroute), AUROC / ECE / equal-mass bin recalibration helpers.

## Tests (std only, deterministic)
* `cargo test --release --offline --lib hybrid` : **14 passed** (tokenizer, zscore incl. constant/empty, top-n ties + NaN, BM25 basics/OOV/empty/dup tokens/no-NaN on empty docs, alpha extremes, zero-variance BM25 (OOV query) and zero query vector, empty corpus + shape errors + NaN/out-of-range alpha, ties + rebuild determinism, serde round-trip with byte-stable serialisation, OSA-1 cases, spell fixes / leaves known+short+no-candidate words, freq-then-lexicographic tie-break, with_known, empty vocab, spell serde, spell-then-search).
* `cargo test --release --offline --lib verifygate` : **17 passed** (AUROC known values incl. ties/NaN/one class, ECE + bins, FixedGate decisions incl. NaN fail-safe, break-even fit incl. ties/NEVER/ALWAYS/empty/NaN items/input order, capture fit incl. <30, calibrate_gate enable + all three fallbacks (WeakSignal on uninformative AND anti-informative, TooFewWrong, BadInput), determinism + serde round-trips (incl. +-1e9 sentinels), ShiftMonitor, PosteriorGate rule/boundary/sums form, evidence rules (Agree / SupportEntity / Latest / SupportAny, Unverifiable), derived claims, poisoned evidence, task assessment, PrecisionGate prior / learning / delay / retry error / serde, EvidenceCritic decisions + serde, order independence).
* full lib suite after the change: **317 passed, 0 failed** (286 before + 31 new).

## PARITY evidence (bins, rerun commands below)
### h16_hybrid (h13 reference: `src/bin/h13_common/{mod,ext}.rs`, fixtures `results/h10_{texts,emb,inst}_{A,B}.*`, `results/h13_picks.json`)
* P1 doc-level: library `HybridIndex` (alpha .7, spell-corrected query texts) reproduces **225** `results/h13_retrieval.json` levels (A tune 1-8, A held 31-60, A confirm 61-90, B confirm 61-90 for `hybz_s`; A confirm for `hybz_raw`; 5 slices x 9 metrics each) with **max |diff| 5.6e-17** (tolerance 1e-12). The 5e-17 is `serde_json` default float parsing of the reference file (1 ulp); the in-process arithmetic is identical. Headline reproduced: A held mix25 MRR .332169 / hit@1 .227899 / F1@3 .182648; A confirm .331870 / .228837 / .182860; B confirm .272579 / .184438 / .146208.
* P2 top lists: `results/h13_top_{A,B}.bin` hybz entries: **42,391 / 42,391 (A) and 16,473 / 16,473 (B)** entries identical (doc ids and dense-cosine f32 bits).
* P3 spell: Rust `SpellCorrector` vs the python `Speller` texts stored in the fixtures: A **33,900 / 33,900** (sp1 doc vocab) and **33,900 / 33,900** (sp2, + query words with query-df >= 3), B **12,892 / 12,892** both; 30,163 (A) / 11,221 (B) of those texts are actually changed by the speller, so the check is not trivial. (First run showed 10 "mismatches": the real-typo family R variant 1 is a human-fixed clean text copied to all three flavours, no speller involved; excluded in the bin, documented in a comment.)

### h16_gate (sources: `src/bin/g9_common/mod.rs`, `src/bin/h3_common/mod.rs`, `src/bin/h11_common/pworld.rs`; 11 g9 critic worlds incl. weak-retrieval, poisoned, report/derived-claim worlds, two uninformative-signal negative controls, two zero-error controls; cal seeds 101-130, test seeds 31-60)
* P1 evidence verifier: library vs g9 `claim_verdict` / `task_verdict` over every claim of 400 tasks x 3 seeds x 5 agents x 4 rules x 3 thetas: **1,224,000 claim verdicts + 792,000 task verdicts, 0 mismatches** (Bad and Unverifiable both well represented, e.g. 29k Bad / 17k Unverifiable in the weak-retrieval world).
* P2 FixedGate: R1/R2/R1cov thresholds vs h3 `calibrate_g` over 11 worlds x 2 verifier prices x 3 call costs: **198 thresholds bit-equal**, **5,940,000** per-record decisions equal, **5,940** `eval_g` runs equal in all fields, AUROC / ECE / recalibration bins **363** values bit-equal.
* P3 precision gate: vs h3 `auto_g` **660 (world, seed) runs exactly equal** (131,659 reroutes in total), and the full g9 `run_critic_arm(Auto)` loop (router, retries, tracker) vs the library `EvidenceCritic` assess -> decide -> reroute -> after_reroute path: **66 runs, all 14 compared accounting fields bit-equal**.
* P4 posterior gate vs the h11 expression `nn < nmin || (1 - s/nn)*rv >= cv`: **664,000 grid points, 0 mismatches**.

## Divergences from the sources (all deliberate, all documented in the module docs)
1. hybrid: `zscore` of a constant vector returns zeros (reference: divides by max(sd, 1e-9), which turns rounding noise of a numerically constant non-zero vector into +-1). Never occurs in the fixtures (parity passes).
2. hybrid: empty corpus / empty query / dimension mismatch handled (`HybridError`, empty result) instead of NaN or panic; `top_n` orders NaN last instead of unwrap-panicking; `alpha` is clamped to [0,1] and NaN -> .7. BM25 average length guarded for empty corpora.
3. hybrid: BM25 postings in a `BTreeMap` (stable serialisation, log-time term lookup; measured latency effect below).
4. spell: ASCII-letter runs only (non-ASCII letters are separators for the vocabulary and kept verbatim); python `.lower()` of non-ASCII letters is not replicated (never hit in the fixtures).
5. verifygate: NaN confidence / NaN posterior means "unknown" and FAILS SAFE to verify (the references compare with `<=` / `>=`, i.e. silently skip). Thresholds are finite sentinels +-1e9 (= reference NEG_INF / POS_INF) so they serialise to JSON.
6. verifygate (intentional new behaviour, h3_proposed_patch P3): `calibrate_gate` refuses to enable the gate when calibration AUROC < .6 or < 30 wrong items and then returns `FixedGate::ALWAYS` (always-verify, today's behaviour). Effect in the h3 worlds (c=.05, tp=.1): enabled in the 6 informative worlds and the weak-signal world (AUROC .68; there R1 itself chooses "always verify"), fallback `WeakSignal` in both uninformative negative controls (AUROC .495 / .504; R1 returned "always" there anyway), and `TooFewWrong` in the two zero-error controls (AUROC undefined). **NEGATIVE: in the zero-error controls h3's R1 returned NEVER (verify nothing, which is right when no draft is wrong) while the guarded library returns ALWAYS (full verifier cost, no benefit).** The guard buys safety against fitting a threshold on noise, not savings; where errors are ~0 and you trust that, call `FixedGate::fit_break_even` directly.
7. `PrecisionGate` owns the delayed-feedback queues (`enqueue_*` + `flush(tick)`) instead of the loop owning them; arithmetic and order are identical (parity P3).

## Benchmark (corpus A = 6,831 real documentation paragraphs, dim 384, SINGLE thread, 240 query texts x 3 timed passes after warm-up; noisy machine: trust ratios, +-50%)
* dense scan + top-8: mean 1658 us (p50 1648, p95 1754)
* hybrid `search` (dense scan + BM25 + 2 z-scores + fusion + top-8): mean **1722 us** (p50 1705, p95 1827) = **+3.8%** over dense-only; BM25 scoring alone 4.0 us; spell `correct()` 3.2 us / query
* build: 37 ms (BM25 index + copy of the dense matrix); spell corrector 39 ms (10,909-term vocabulary)
* bytes: dense matrix 10,492,416 (10.49 MB); BM25 payload **1,032,689 (1.03 MB, +9.8% over dense)** (matches h13's +1.0 MB); `serde_json` of the BM25 index 1.63 MB; of the spell corrector 0.24 MB
* The 3-8 ms query ENCODE (MiniLM) dominates anyway; no extra model call. h13 reported a net +23 us (+1.4%) for fusion; this implementation measures +64 us (+3.8%) because it recomputes the dense scan inside `search`, uses a `BTreeMap` term table and includes the BM25 scoring; still negligible against the encode.

## Honest caveats
* The retrieval gain (+.03..+.05 MRR over dense+spell) is a doc-level effect on long-tailed real documentation; on templated/paraphrase corpora it is ~0 and in the lowest lexical-overlap tertile it is negative (h13). End-to-end routing ties. Ship as opt-in, keep dense(+spell) selectable (`alpha = 1.0`).
* The evidence verifier only pays where retrieval surfaces the current document >= ~65% of the time and first-draft error >= ~5% (g9); the h3 fixed gate saves verifier calls only when the confidence signal is informative and the test error rate is near the calibration error, and does NOT beat online learned stopping in orchestration (h3). Posterior-gated verification cut cost ~27% at equal quality only where verification is a large part of the bill (h11). All of these are simulations with synthetic signals.
* Nothing is wired into `ResearchConfig` (`critic`) or `KnowledgeConfig`; that integration is the director's call (see `results/g9_proposed_patch.md`, `results/h3_proposed_patch.md`, `results/h11_proposed_patch.md`, `results/h13_proposed_patch.md`).

### VERIFY REQUEST — R8-A — h16 unit tests
```
export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-rN RAYON_NUM_THREADS=2
cargo test --release --offline --lib hybrid       # expect: 14 passed
cargo test --release --offline --lib verifygate   # expect: 17 passed
cargo test --release --offline --lib              # expect: 317 passed (286 + 31)
```
Check that the tests assert behaviour (determinism, serde round-trip with byte-stable output, edge cases, guards), not just compilation: mutate `FixedGate::should_verify` or the guard in `calibrate_gate` and confirm tests fail.

### VERIFY REQUEST — R8-A — h16_hybrid parity + benchmark
```
cargo run --release --offline --bin h16_hybrid     # ~2.3 min (H16_THREADS=2 for the parity part), writes results/h16_hybrid.{json,txt}
```
Expect `PARITY 1 PASS: 225 levels, max |diff| ~5.6e-17`, `PARITY 2 PASS` (42391/42391 A, 16473/16473 B), `PARITY 3 PASS`, hybrid overhead vs dense scan single-digit percent, BM25 payload ~1.03 MB (+9.8%). Non-zero exit code if parity 1 or 2 fails. Cross-check by eye: `results/h13_retrieval.json` `held_A_31_60.levels.hybz_s.mix25.Mrr` = .332169 vs `results/h16_hybrid.txt`.

### VERIFY REQUEST — R8-A — h16_gate parity
```
cargo run --release --offline --bin h16_gate      # ~5 s, writes results/h16_gate.{json,txt}
```
Expect `P1..P4 PASS` and `OVERALL PASS`: 1,224,000 claim verdicts + 792,000 task verdicts equal, 198 thresholds bit-equal, 5,940,000 decisions equal, 660 `auto_g` runs + 66 full `run_critic_arm(Auto)` loops equal, 664,000 posterior-gate points equal; the guard table lists which worlds enable / fall back (divergence 6, including the zero-error-control NEGATIVE).

## Summary
Two std-only opt-in library modules (`hybrid`, `verifygate`) with 31 unit tests (317 total pass), bit-level parity with the h13 retrieval fixtures (225 levels, 58.9k top lists, 46.7k spell texts) and with the g9 / h3 / h11 gating code paths (about 9M compared decisions/values, 0 mismatches). Hybrid retrieval costs +3.8% per query and +9.8% bytes over dense on the 6,831-doc real corpus. Defaults unchanged, nothing wired in. Known behavioural negative of the new guard: it converts "never verify" into "always verify" in zero-error regimes.
