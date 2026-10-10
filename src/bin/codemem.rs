//! Stand-alone launcher for codemem (same source as `neural_mesh::codemem` / `neural-mesh codemem ...`, compiled directly into this bin via
//! `#[path]` so it links only serde, serde_json and regex: a much smaller exe, which matters for process start-up in the UserPromptSubmit hook).
//! Usage: `codemem <query|update|status|hook|inject|bench|eval|add> ...`.
#![allow(dead_code)]
#[path = "../codemem/mod.rs"]
mod codemem;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let is_hook = args.first().is_some_and(|a| a == "hook");
    let code = std::panic::catch_unwind(|| codemem::cli::run(&args)).unwrap_or(if is_hook { 0 } else { 3 });
    std::process::exit(code);
}
