# H7 world definitions frozen (written BEFORE any H7 policy / arm was run)
timestamp (UTC): 2026-10-04T22:33:08Z
src/bin/h7_common/hworld.rs  sha256 285e108b9e03ebe15790f9527fee3d6492594ca25870c32f015c882c547b72ef
src/bin/g4_common/world.rs   sha256 ea287f5903554394f24bd3e47a2061cac7b84d964d4363c843d2188ecfb36224   (W4 frozen generator, reused untouched via #[path])
New generators N1 grid/QD-archive, N2 global-corpus, N3 modal-concave, N4 grid-recent24 (unattainable), NK1-3 controls, NM1-2 mixed: see hworld.rs header.
Disclosure rule: any later edit to hworld.rs is listed in results/h7_board_entry.md under PROCESS DISCLOSURE.

AMENDMENT 1 (2026-10-04T22:38:03Z, still BEFORE any H7 policy ran): rustc edition 2024 reserves the identifier `gen`; renamed field/enum-member access `gen` -> `gk` in hworld.rs (mechanical, no semantic change).
src/bin/h7_common/hworld.rs  sha256 80e2071cb4bde99b2746d19641b9084b54564094eb45f4b56ebed592a18e35e6   <- the hash to verify
Engine validation (exp validate, run before any experiment): h7 engine with k=1 reproduces g4 run() EXACTLY (max |diff| 0) on 8 W4 worlds x 10 seeds x {fair x3, W4 nbucb/nbreg, NbX reimplementations}.
