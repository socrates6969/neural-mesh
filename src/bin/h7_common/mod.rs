//! H7 (round 4, creativity widening). Reuses the FROZEN g4 world + policies through #[path] (no copy, no edit).
#![allow(dead_code)]
#[path = "../g4_common/world.rs"]
pub mod world;
#[path = "../g4_common/pols.rs"]
pub mod pols;
pub mod hworld;
pub mod engine;
pub mod pols2;
