//! g21 shared pieces (W22, R&D round 3, TOOLS round 2): natural-language task text -> spec EXTRACTOR in the loop,
//! an INDEPENDENT verifier, and seeded evaluator bugs (mutants). Reuses the g13 world (agents, error curves, task
//! mix, canonical specs, ground truth) unchanged via `#[path]`; `src/*.rs` is untouched.
#![allow(dead_code)]
#[path = "../g13_common/mod.rs"]
pub mod g13;
pub mod evalx;
pub mod extract;
pub mod indep;
pub mod nlgen;
pub mod pipe;
