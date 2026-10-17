# H12 proposed patch
Nothing to patch in src/. The layers live in src/bin/h12_common/online.rs and are opt-in by construction (defaults unchanged).
Only if the director wants a score-space residual (instead of the behavioural override used here), the Router trait would need an optional `fn scores(&self, task:&Task) -> Option<Vec<f64>>` on Warm/Guarded (HebbianRouter already has `scores`). Given the verdict (not kept; gains only in interaction worlds) I do NOT recommend adding it.
