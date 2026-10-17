# h3 proposed patch (NOT applied; I own no src files). Opt-in only; defaults unchanged.

Context: results/h3_gate_board_entry.md. A calibrated, FIXED-threshold confidence gate in front of a verifier cut verifier calls substantially in the simulated worlds, but only when
(a) the confidence signal is informative, (b) test-time error rate is close to calibration, and (c) there is no already-good online stopping rule (g18 L4) to compare against.
Because (c) failed in the orchestration world (the existing L4 online stopping already matches or beats the calibrated gate), the recommendation is a small opt-in primitive, not a default.

## P1 `src/research.rs` (or a new `src/gate.rs`): `FixedGate`
```rust
/// Verify iff `score <= threshold`. The threshold is data (fitted offline on a calibration split with ground-truth outcomes), never updated online.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct FixedGate { pub threshold: f64 }
impl FixedGate {
    pub fn should_verify(&self, score: f64) -> bool { score <= self.threshold }
    /// R1 (cost-neutral break-even): items = (score, net_benefit_of_verifying) on the calibration split; returns -inf (never verify) when no prefix has a positive net benefit.
    pub fn fit_break_even(items: &[(f64, f64)]) -> FixedGate { /* sort by score, maximise the cumulative benefit at tie boundaries; see r1_threshold in src/bin/h3_common/mod.rs */ }
    /// R2 (target error capture): the `cap` quantile of the scores of WRONG items; verify nothing when < 30 wrong items were observed.
    pub fn fit_capture(wrong_scores: &[f64], cap: f64) -> FixedGate { /* r2_threshold */ }
}
```
Reference implementation + unit tests of both rules: `r1_threshold`, `r2_threshold`, `auroc_low_is_pos`, `ece`, `fit_bins` in `src/bin/h3_common/mod.rs` (tests in `src/bin/h3_gate.rs`).

## P2 `ResearchConfig` / critic path
Add `critic_gate: Option<FixedGate>` (default `None` = current behaviour) and a `confidence: Option<f64>` field on the executor's draft output; when set and `critic` is on, skip the critic call if `!gate.should_verify(confidence)` and count it as a skipped call in the report.

## P3 diagnostics to ship with it
A `calibration_report(scores, wrong)` helper returning AUROC, raw ECE and recalibrated ECE; refuse to enable the gate (return `None`) when calibration AUROC < 0.6 or the calibration split has < 30 wrong items. Re-calibrate (offline) whenever the observed error rate on a monitoring window leaves a pre-declared band around the calibration error rate (the shift tests show the gate is fragile when the test error rate is higher than calibrated).

Not proposed: any online threshold adaptation (rejected in earlier rounds), enabling by default.
