# g21 proposed patch (W22) - NOT applied: `src/*.rs`, Cargo.toml, lib.rs are untouched

All code below already exists, tested, in `src/bin/g21_common/` (std-only, no new crate). Numbers are SYNTHETIC-until-validated-on-real-traffic (see results/g21_board_entry.md).

## 1. `tools::IndependentVerifier` (from `src/bin/g21_common/indep.rs`, ~700 lines)
Why: W14's `Verifier` re-uses the same `Calc`/`Units`/`Csp`/`Identity` code as the tools, so a parser, unit-table or csp bug is invisible to it (mutation test: 0.0% of silent wrong outputs caught for 12 of 14 live bugs, 19% of all; the independent path catches 100% of every non-common-mode bug, 90.0% overall; 0 false rejects on 4384 canonical + 9000 random expressions; differential test of the real `tools::Calc` found no bug).
Sketch (pub API, implements `Tool`, same 3-line input `tool\nspec\nclaim`, same `accept` / `reject: why` output):
```rust
pub struct IndependentVerifier { lookup: LookupTool, shared: Verifier, pub meta: bool }
impl IndependentVerifier {
    pub fn new(lookup: LookupTool, meta: bool) -> Self;
    /// (verdict, cost, used_independent_path); falls back to `shared` only for unsupported inputs
    pub fn verify(&self, tool: &str, spec: &str, claim: &str) -> (Result<String, ToolError>, u32, bool);
}
```
Pieces: shunting-yard -> RPN + own `Fr` (checked i128 fraction) + outward-rounded f64 `Iv` interval (claim must lie in it); unit table as DECIMAL STRINGS (ft "0.3048", lb "0.45359237", knot "1852/3600") with its own dimension vectors; csp certificate check + brute-force "unsat" re-solve (<= 400k assignments); identity grid on the independent evaluator; native re-implementation of the six program templates (charge the verifier like the interpreter re-run to stay conservative); optional metamorphic re-print (fully parenthesised and operand-commuted parse sent to the tool under test; adds +0.45 cost/task end to end and no catch beyond the independent evaluator in the mutation suite - recommend `meta = false`).
Recommended wiring: `Registry::with_defaults` registers `IndependentVerifier` at index 6 instead of `Verifier`, or `Verifier::verify` returns `reject: evaluators disagree` when the two paths differ. Cost: +0.03..+0.09 per task vs the shared verifier; correctness effect on a correct evaluator +0.0002 [-0.0001,+0.0005] (nil); under a real bug +0.3..+2.7 pp end to end.
Documented limit to put in the doc comment: independence = different algorithm and encoding written by one author; a bug planted in BOTH paths is caught 0% (rows CM_U_lb, CM_P_addsub_right).

## 2. `tools::spec_extract` (from `src/bin/g21_common/extract.rs`, ~1500 lines)
```rust
pub struct Extraction { pub spec: Option<String>, pub conf: f64, pub kind: Option<ToolKind>, pub n_cands: usize, pub fuzzy: usize, pub unused: usize }
pub fn extract(text: &str) -> Extraction;   // deterministic, std-only; spec = None means ABSTAIN
```
Contract to document: confidence >= .9 was right 99.0-99.8% of the time on the synthetic worlds, .5-.7 only 69-75%; semantic accuracy 96% / 91% on phrasing it was built for and 52% / 39% on phrasing it was not (worlds A / B); it must be re-validated on real task text before any threshold is trusted. Output is FULLY PARENTHESISED on purpose (it hides evaluator precedence/associativity differences; keep that property, or keep the independent verifier on).
Usage with the existing selector: features = `tool_features(spec)` + conf bin token (`cf:lo|mid|hi|top` at .3/.6/.8) + kind-guess token; BUT the measured result is that a one-parameter rule (use the extractor's tool iff conf >= 0.4) beat the learned selector by 1.2-1.9 pp, so ship the rule first and the learner only if it earns its keep on real traffic. The rule fails the all-prose control (-0.8..-3.3 pp, TOST +-1pp false), so pair it with a prose guard (discussion-verb list is in `extract()`; coverage of unseen prose is weak).

## 3. `ToolSelector` reward warning (doc comment only)
The g13/W14 reward "verified accept = 1.0" is a fine proxy when the SPEC is known right; with an extractor in front, a wrong spec still passes verification, so the reward must include ground truth (or a spec-level check) or the selector is trained on false positives (measured: indep - noverify = -0.006 / -0.008 in world B).

## 4. Not proposed (measured not worth it)
* metamorphic-only verification (catches 55% of parser-class bugs, over-rejects 26771 correct integer-division outputs on the stress corpus);
* a larger unit table copy in the verifier (same constants typed twice add no independence for factor mistakes made from the same source).

## 5. Suggested follow-ups (unmeasured)
* second, differently-built extractor (or the agent's own formulation) as a spec-level cross-check: the one place where verification could regain the +0.10 W14 attributed to it, because today nothing can see an extraction error;
* on a verifier reject with an independent value available, adopt the independent value instead of falling back to an agent's direct answer (recovery in the bug scenarios is partial, 65-75% right).
