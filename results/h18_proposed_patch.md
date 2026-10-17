# h18 proposed patch (no src/tools.rs edit made; src/lib.rs got exactly one line: `pub mod fin;`)
Optional, for the director: register the domain tools next to the W14 defaults so a Registry carries them. In `tools::Registry::with_defaults` (NOT done by me):
```rust
r.register(Box::new(crate::fin::FinTool::default()));   // index 7
r.register(Box::new(crate::fin::SciTool::default()));   // index 8
r.register(Box::new(crate::fin::EconTool::default()));  // index 9
r.register(Box::new(crate::fin::DomainVerifier { indep: true, bug: 0 }));
```
Caveats: the evaluator bug hooks (`fin::bug`) are compiled in for mutation testing (0 = production; unit test `bug_hook_zero_is_production`); a deployment that wants zero hook surface can gate them with `#[cfg(feature = ...)]`. Use the extractor only behind a confidence gate (conf >= ~0.7): ungated use is harmful on prose and on unseen phrasing (see results/h18_board_entry.md).
