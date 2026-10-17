# h21 proposed patch (PROPOSALS ONLY; no src edits were made; defaults unchanged; all synthetic-until-validated)

## P1 (src/verifygate.rs, docs + tiny constructor, no behaviour change)
`PosteriorGate { rv, cv, nmin }` is correct as shipped (h21 links it unchanged and a unit test shows rv=1e9 reproduces always-verify bit for bit), but the tuned `rv` is not a free learned parameter: it is the exchange rate, rv ~ 1/mu (quality worth 1/mu reference calls). On the h21 mixture at mu .03 the objective is a plateau for rv in 8..32 (always-verify unless m >= ~.99), at mu .1 rv = 4. Suggest (a) a doc line saying so, and (b) an optional constructor `PosteriorGate::from_exchange_rate(mu, cv, nmin) -> rv = 1/mu` so callers state mu instead of tuning rv. Expected effect: none on defaults; avoids the "tuned gate never fires / always fires" confusion seen in h11 (world D rv* = 40) and h21 (rv* = 32).

## P2 (research.rs / dispatcher: DO NOT enable cross-source READ verification by default)
The only component of the pre-declared h21 stack that HURT was cross-source READ verification (-.035..-.065 ok, +.4..+.9 calls vs the span check); keep the per-step local checks (source-span, tool re-computation, independent re-query) and per-step title check on searches. `verifygate::EvidenceRule::Agree` (g9) is a per-claim cross-source rule; h21 suggests using it only where the retrieval surfaces the current document >= 65% and the SEARCH step is NOT already verified.

## P3 (new, untested here: abstain / memory fallback)
h21's `ctl-noinfo` shows retrieval-based pipelines are WORSE than memory-only when the corpus lacks the answer (-.04..-.09 q at up to 2.7x cost) because nothing abstains. Proposal for a future study (not implemented, no claim): after two flagged/failed searches for a step, stop retrieving, fall back to parametric memory or mark the step UNVERIFIED and skip dependent computation. Needs its own world (coverage < 1) and a control with full coverage where the fallback must not fire.

## P4 (tools.rs): none required
`Calc` was used as the exact compute tool (12k random specs agree with native arithmetic; operator-mutated specs evaluate and differ), no change needed.
