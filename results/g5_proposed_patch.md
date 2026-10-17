# G5 proposed library changes (NOT applied; W5 owns only src/bin/g5_* and results/g5_*)

Nothing in G5 needed a src change to run. These are the changes I would want if the G5 findings are accepted.

1. `research::ResearchSession` takes its router from `CoupledBrain` only (`warm` or `guarded(hebr)`), so the
   domain-hierarchical router (`HierRouter`, src/bin/g5_common/mod.rs) cannot be used inside the verified pipeline.
   G5 therefore runs `hier+retr` in an in-bin replica of `run_task` (critic = None, reuse = true, health off). The replica is
   checked bit-identical to the real session for the safe and warm routers, with and without retrieval, including the
   brief-ingest path (`g5_study --check`), but `hier+retr` itself is not library code. Proposal: let the session accept any
   `Box<dyn Router>` front (keeping `CoupledBrain` as the default) or a generic `ResearchSession<R: Router>`.
2. `research::Context` exposes `anchor`, `note`, `hits` but not the best cosine or the number of distinct records behind the
   anchor. In G5 the only gate against a wrong-project brief is the global `sim_min`; the wrong-brief harm (design -0.011..-0.025
   in `low-recurrence` / `confusable-briefs`) comes from anchors that mix projects. Proposal: add `top_sim: f32` and
   `anchor_size: usize` to `Context` so an executor/agent can discount low-confidence notes, and optionally store a caller-supplied
   `scope` key (project/client id) with each note so retrieval can be filtered by metadata instead of text similarity alone.
3. (Optional, opt-in) move `HierRouter` into `src/` next to `prior.rs`: 3-level shrinkage (agent -> domain -> sub-skill) with a
   keyword pseudo-count. G5 shows the domain level is worth +0.005..+0.028 over the same router without it (all 10 worlds, both
   seed sets), but only +0.005..+0.014 over tuned `warm(hebr)` at H1500 and it is behind for the first ~100 tasks, so I do NOT
   propose changing any default.

No change to `defaults::recommended_router` is proposed. In the G5 graded worlds `warm(hebr)` beat `guarded-safe` by 0.01-0.02
(see board entry); that is a world-dependent observation and f7 found the opposite sign in drift/stress worlds.
