# G18 proposed patch (W19). NOT applied: src/*.rs is untouched. Everything below is a design proposal backed only by the simulation in `src/bin/g18_common/mod.rs`.

## 1. `research.rs`: a step-level entry point next to `run_task`
`ResearchSession::run_task(task, executor)` routes one whole task to one agent. The G18 orchestrator needs three things the session does not expose:
- `run_plan(steps: &[PlanStep], executor)` where `PlanStep { ty: String, deps: Vec<usize>, project_key: Option<String> }`; route each step with `task_type = ty`, feature = ty (what `g18` calls a `stask`), and call `executor.execute(step, agent, &Context, upstream: &[Draft])`.
- retrieval keyed per (project, step type) rather than per whole task (`Knowledge::retrieve` with the step text, anchor per step). In G18 this is the single biggest generic lever (L2 - L1 = +.08 graded in generator A, +.067 in B at the tuned setting), and it is the part that needs the real embedding index, which G18 only emulates with a stochastic hit model (p_id, wrong-note penalty).
- `ResearchSession::with_router(Box<dyn Router>)` (G5 already asks for this; it is the reason `g5_study` re-implements `run_task`).

## 2. new `VerifyGate` (pure policy, no randomness besides one exploration draw)
```rust
pub struct VerifyGate { stats: HashMap<(usize /*agent*/, String /*step type*/), (f64, f64)>, pub r_val: f64, pub eps_ex: f64 }
impl VerifyGate {
    /// verify iff ucb(flag rate of this (agent, type)) * r_val >= cost_of_one_check, or with prob eps_ex (exploration)
    pub fn should_verify(&self, agent: usize, ty: &str, check_cost: f64, u: f64) -> bool { ... }
    pub fn observe(&mut self, agent: usize, ty: &str, flagged: bool) { ... }
}
```
ucb = p + sqrt(p(1-p)/(n+3)), p = (flagged + .5)/(n + 2.5). Tuned values in G18 (seeds 1-8, objective = utility at mu .03): `r_val = 1.0`, `eps_ex = 0.1`. The `eps_ex` floor means ~0.1 verifications per step are paid forever (see `L4-noexplore` row in the logs: it stops verifying entirely, but cannot recover if the verifier or an agent drifts; drift was NOT tested in G18).

## 3. contract for a verifier / reroute
- `Verifier::check(step, draft) -> Signal(f64 in [0,1])`; flag iff signal < theta_v (G18: .55). Never accept a flagged draft silently: on flag, reroute to the best router choice among UNTRIED agents (max 2 retries), keep the highest-signal draft if all fail.
- Feed the verification signal back to the router immediately (`router.update(step_task, agent, signal)`); G18 ablation `L3-nolearn` shows this is worth +.011 graded (mean over generator-A worlds: +.0110 on 31-60, +.0104 on 61-90; +.002..+.003 in generator B and controls).
- A verifier with signal noise >= ~.22 (same scale as the quality threshold gap) makes always-verify HURT (G18 sensitivity: sig_v=.22 -> L3-L2 = -.005 graded, .4 -> -.048). Ship `VerifyGate` only together with a measured verifier ROC; a gate that learns from its own flags cannot detect a verifier that flags random good steps (it will keep verifying).

## 4. what NOT to put in the library on this evidence
Parallel fan-out (L5): quality-neutral by construction here (no model of shared-resource contention or of rate limits); only a scheduling helper (`critical_path(steps)`) is justified. The latency numbers (-0.33 steps in pipe-main, -2.1 in lat-main) depend entirely on how many steps have no dependencies.
