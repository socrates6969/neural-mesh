# G8 / W9 proposed patch

None for the spiking mechanisms: rejected (see results/g8s_board_entry.md). No src/*.rs file was edited; the implementation lives entirely in src/bin/g8_snn.rs and src/bin/g8_common/.

Optional, only if the director wants to pursue the NON-spiking control (rate-ctl) as a separate research item: it is `Learner` in src/bin/g8_common/snn.rs with `Sel::Rate`. A src/ module would need (a) `Learner`, `Pop`, `LCfg`, `Credit` moved to e.g. src/rbfdelta.rs with the `Router` trait adapter (Task features -> token slots, no continuous x in the current Router trait, so population coding cannot be used through `Task` without adding an `x: Vec<f64>` field to `Task`, which is a signature change the DESIGN.md forbids), and (b) a cross-world single-config evaluation first. Neither is recommended before that evaluation: the rate-ctl margin over matched-effort baselines is only about +0.008 absolute and appears in the static control world as well.
