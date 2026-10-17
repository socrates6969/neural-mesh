# G9 (W10) proposed library patches (NOT applied: src/*.rs untouched)

Everything below is a design sketch justified only by the simulations in `results/g9_board_entry.md`. Nothing here should ship without a re-test on real text / real agents.

## P1. Evidence-aware critic hook in `research.rs` (the only critic design that earned its cost)

Measured: the shipped gate (`CriticKind::Flat/Hier`, second agent WITHOUT evidence) is a ~no-op in `g9_critic` (it checks ~1% of tasks, net -0.0003..-0.002 vs none);
a verifier that reads retrieved evidence (cross-source agreement) wins +0.054..+0.071 net@0.05 where retrieval surfaces the current document >= ~65% of the time and
errors are >= ~6%, and loses where retrieval is weak (p_ret <= 0.5) or evidence is poisoned/single-sourced. So the hook must be opt-in and self-disabling.

```rust
// research.rs
pub enum Verdict { Ok, Bad, Unverifiable }

pub trait Executor {
    // ... existing methods ...
    /// Evidence check of a draft against what retrieval returned for this task (tool run, NOT an agent call).
    /// Default: Unverifiable, so every existing executor keeps its behaviour.
    fn verify(&mut self, _task: &Task, _out: &Self::Out) -> Verdict { Verdict::Unverifiable }
}

pub enum CriticKind {
    Flat(SecondOpinionConfig),
    Hier(HierSoConfig),
    /// NEW: evidence verifier + reroute on Bad, with an online precision gate.
    Evidence(EvidenceCriticConfig),
}

pub struct EvidenceCriticConfig {
    /// price of one other-agent call in success units (used for the break-even only)
    pub call_cost: f64,            // g9 default 0.02..0.05
    /// treat Unverifiable as a flag? (g9: false is better in every tune world)
    pub flag_unverifiable: bool,
    /// prior pseudo-count for "reroute error = first-draft error + 0.05"
    pub retry_prior_n: f64,        // 8.0
}
```
Run-loop change (inside `run_task`, replacing the SecondOpinion branch when `CriticKind::Evidence`):
1. `v = ex.verify(task, &first)`; count it as a tool run in `SessionStats` (new field `verifies`), NOT in `calls`.
2. if `Bad`: act only if `P(first wrong | flagged) > P(retry wrong) + call_cost*(1+2*tool_cost)`, both estimated online from the delayed graded outcomes the session already receives
   (`apply()` has `g_first`, `g_retry`): Beta(1,1) for the flag precision, retry error prior = running first-draft error + 0.05 with `retry_prior_n` pseudo-counts. If not acting, deliver the draft marked unverified.
3. act = one retry by the best OTHER agent (existing `plan.retry`), verify the retry, deliver it if it passes, else the draft with fewer Bad claims, marked unverified.
4. `TaskReport` gets `verdict: Verdict` and `delivered_unverified: bool` (a visible flag is useful even when no rerouting is done: g9 silent-error rate 0.103 -> 0.007 in A1).
Why the online gate: in `A11` (single source, half the facts poisoned) the ungated verifier is -0.040 net; with the gate -0.010 (the gate stops acting after ~100 graded flags); in the zero-error controls the gate halves the cost (-0.0066 vs -0.0075 A10, -0.0065 vs -0.0136 B3).

Default stays `critic: None`. Documentation line to add: "enable `Evidence` only if the retrieval hit rate of the CURRENT document is >= ~0.65 and the first-draft error rate is >= ~0.05; otherwise leave it off".

## P2. Hybrid (dense + tf-idf) similarity and spell-correction for `knowledge.rs`

Measured (g9_sample, frozen g1 picks, fresh corpora C/D): the knowledge layer's retrieval is the bottleneck of sample efficiency (half of the headroom to the oracle needs ~1000-1500 tasks) and
a dense+tf-idf blend (alpha .6-.7) reaches that point ~350-450 tasks sooner than BGE-alone in the sibling-heavy world (sib16, Holm WIN on both fresh corpora) but NOT in rand16 / sib64.
```rust
// knowledge.rs
pub struct KnowledgeConfig {
    // ...
    /// 0.0 = dense only (default, unchanged). Otherwise sim = alpha*dense + (1-alpha)*tfidf_cosine over the stored task texts.
    pub lexical_blend: f32,
    /// idf table is built online from stored task features (no persisted state needed)
}
```
and an `Embedding` decorator `SpellFixed<E: Embedding>` that snaps each token with edit distance 1 to the store vocabulary (tokens with count >= 3) before embedding.
Both are strict opt-ins (default = bit-identical to today). The tf-idf side needs only the token features that `Task` already carries; no new crates.

## P3. Constraint pinning for multi-step sessions

Measured (g9_multistep): in a session whose later queries are written from earlier evidence, the detail slot is the failure channel: a wrong detail in step 1 is silently carried (dependent-session success
vs the same system with the gold detail restated: see board entry). Verifying the retrieved document's attribute against the attribute the query states (`local`) and, additionally, keeping the USER-stated attribute
from step 1 (`pin`) instead of the evidence-inherited one are the two cheap levers.
```rust
// research.rs
pub struct SessionConstraints { pub pinned: Vec<(String /*slot*/, String /*value*/)> }
pub trait Executor { fn attrs(&mut self, _text: &str) -> Vec<(String, String)> { vec![] } }   // gazetteer / extractor, default none
impl ResearchSession {
    pub fn run_chain<E: Executor>(&mut self, ex: &mut E, steps: &[Task], candidates: &[usize], rng: &mut Rng) -> Vec<TaskReport<E::Out>>;
    // step 1: pin attrs(user query); every later step: ctx gets `pinned`; a retrieved note/doc whose attrs contradict a pinned slot is demoted below the first non-contradicting candidate
}
```
`Context` gets `pub pinned: Vec<(String, String)>`. All default-empty, so the single-task path is unchanged.

## Not proposed
* Self-consistency voting as a default critic: costs 2 extra calls on every task; net-negative in every world at call cost c >= 0.05, and at c = 0.02 it only wins where errors are random (A3 halluc-heavy +0.028, A4 arith-heavy +0.018, A6 random-slips +0.042 vs none, 31-60 means) and is ~0 or negative elsewhere.
* Evidence substitution (deliver the evidence value instead of rerouting): needs retrieval to be right; it is the worst arm wherever retrieval is weak or poisoned (A7 -0.14, A8 -0.11, A11 -0.35).
