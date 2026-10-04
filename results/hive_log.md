# Hive log (queen = nested-queen; brain = neural-mesh CLI, state .claude-flow/hive/state.json)
## Round 1 routing (cold brain, all scores 0.6 tie)
| task | features | brain pick | scores | final assignee | override? |
|---|---|---|---|---|---|
| T1 router/warm-start | hebbian,router,warm-start,prior,efficiency,keyword | mesh-researcher | 0.6/0.6/0.6 | router-researcher (W1) | YES: tie, task clearly router specialist |
| T2 mesh/edges | mesh,edges,workflow,fault-tolerance,sparse,handoff | sona-researcher | 0.6/0.6/0.6 | mesh-researcher (W2) | YES: tie, clearly mesh specialist |
| T3 SONA | sona,ruflo,port,benchmark,research,learning | sona-researcher | 0.6/0.6/0.6 | sona-researcher (W3) | no |
## W1 round 1 (router-researcher): ArmWarm (Thompson meta-bandit keyword vs learner) in prior.rs + w1_lab.rs
Outcome: REJECTED on bar. Main world no sig win (H1000 sig LOSS vs warm(hebr), -0.016 Holm p=.041); correct-keywords sig loss to keyword all horizons; many-topics H500 win +0.0129 (Holm p=.004) isolated. ns/route: kw 140, hebr-robust 1448, warm 2445, ArmWarm 1667. Tests not verified in real crate (mesh.rs mid-edit); scratch run 91 pass/1 fail (sona.rs test, W3's).
Reward 0.5 (useful negative). Brain feedback recorded.
## W3 round 1 (sona-researcher): src/sona.rs + w3_sona.rs
Outcome: no variant passes bar. Faithful SONA = keyword + lock-in, no exploration, Q stage 0% of picks (native engine dead code); sig worse than keyword (-0.014) and warm(hebr) (-0.05..-0.14). Best variant sona-feature(+explore) beats keyword from H500 but loses to warm(hebr) at all horizons. Lib tests 46 pass (real crate, per W3).
Reward 0.5 (useful negative; port + documentation valuable).
## W2 round 1 (mesh-researcher): SeqEdges/SeqAdd (variants.rs), FaultConfig (mesh.rs), w2_workflow, w2_fault
Outcome: all negative. 0/30 sequential tests beat warm(hebr) (main: -0.018..-0.036); 0/32 fault tests beat best baseline; streak-bench harmful (0.55 vs 0.76). Sparse edges only save memory at >=50 agents (5.6x @50, 56x @500). Reward 0.5.
## Lead verification
cargo test --release --lib: 110 passed, 0 failed (real crate). Reran w3_sona: ACCEPTED [] and faithful SONA numbers reproduce (0.651/0.642/0.640/0.640/0.640).
## Final brain state
All three rewards were 0.5 == the EMA baseline, so Hebbian delta ~ 0: scores still tie at 0.594 for all agents; ties broken to mesh-researcher. Routing did NOT learn to match specialists; no follow-up round run (time budget).
