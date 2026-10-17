# h19 proposed patch (R9-C) -- PROPOSAL ONLY, no src/*.rs was edited

Nothing here is applied. All h19 code lives in `src/bin/h19_*.rs` + `src/bin/h19_common/` and uses only existing lib APIs
(`router::{Router, Rng, Task}`, `baselines::*`, `hebbian`, `prior::Warm`, `defaults::recommended_router`, `hybrid::Bm25`, `stats::*`).

## What could be promoted (opt-in, defaults unchanged)

1. `src/docextract.rs` (new, std-only, ~900 lines, copy of `h19_common/{scan,verify}.rs` minus the generator-specific lexicons):
   * `split_cells`, `parse_num` (parentheses/minus/currency/%/K-M-B, US vs EU separators with an explicit `ambiguous` note),
   * `Ext { fact, conf, notes }` with per-field confidence and an abstention threshold `tau` (calibrate per corpus; the h19 default 0.5 is a
     coin flip, not a calibrated value),
   * `verify_doc(...)` returning `Vfy { reasons, repair }`: independent span re-parse, independent scale evidence, plausibility priors,
     accounting-identity footing with culprit localisation, redundancy cross-checks, claim-coverage check.
2. Register two tools in `tools::Registry`: `table_extract(text) -> facts+conf` and `verify_facts(text, facts) -> flags+repairs`, so the
   existing exact calculators (`tools::Calc`) can consume verified values. Keep the verifier a separate call (independence matters:
   a verifier that shares the extractor's lexicon shares its blind spots -- see board entry, "narrow verifier" ablation).
3. In `verifygate.rs` the `FixedGate`/`PosteriorGate` machinery could gate the *span re-read* (the expensive step) on the verifier's
   `reasons` mask instead of a scalar score: reasons {span, sign, scale, foot} had near-zero false-flag rate in h19; {claim_coverage,
   foot_ambiguous} are the noisy ones.

## What NOT to promote
* The generators / unseen-bank lexicons (they encode the author's idea of "unseen"; an extractor tuned on them would be circular).
* The reader-noise constants in `down.rs` (assumptions, not measurements).

## Tests that would be needed before any promotion
* real PDF-to-text fixtures (h19 never saw PDF extraction artefacts: ligatures, column interleave, hyphenation, rotated tables),
* a labelled real corpus (h19's real-text check has no ground truth),
* a calibration set that contains WRONG extractions (the dev bank has none, so a dev-calibrated tau is meaningless: tau* = 0.05 at F1 1.000).
