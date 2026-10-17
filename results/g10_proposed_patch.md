# g10 proposed lib patches (NOT applied; src/*.rs untouched). W11, R&D round 3.

Both patches are small and opt-in. Neither is recommended as a default; the measured evidence is in results/g10_board_entry.md.

## P1 (E1): self-calibrating `sim_min` in `knowledge.rs`  -- recommendation: DO NOT ADOPT the specified rule; the background-floor variant is an optional opt-in only

Measured: the specified rule (running Welford mean - kstd*std of the top-1 cosine, bootstrap 0.6 for 30 queries, one frozen kstd) is NOT
TOST-equivalent (+-0.01) to the per-embedder tuned `sim_min` for anchor pair-F1 in any of the 18 held-out cells (and any of the 18 confirm cells); the
frozen kstd=0.5 recovers none of the tuned-vs-default gain. The variant "threshold = mean + z*std of the cosine between the query and
4 pseudo-random earlier queries (chance level of the embedder), z=2 frozen" recovers about half of that gain (anchor F1 +0.054 vs the library default
sim_min .6, k5) but is still far from equivalent to tuning.

If the director still wants the opt-in, the minimal change is (all in `Knowledge`, no new dependency):

```rust
// knowledge.rs
#[derive(Clone, Debug, PartialEq)]
pub enum SimMin {
    Fixed(f32),                                   // today's behaviour (default, bit-identical)
    /// threshold = max(boot until `n_boot` queries seen, bg_mean + z * bg_std) of the cosine between each query and
    /// 4 deterministic pseudo-random earlier records (Welford, causal)
    Chance { z: f32, n_boot: usize, boot: f32 },
}
pub struct KnowledgeConfig { /* ... */ pub sim_min: f32, pub sim_rule: SimMin }   // new field, default SimMin::Fixed(sim_min)

pub struct Knowledge { /* ... */ bg: Welford /* n, mean, m2 */ }

fn thr(&self) -> f32 {
    match self.cfg.sim_rule {
        SimMin::Fixed(s) => s,
        SimMin::Chance { z, n_boot, boot } =>
            if self.bg.n >= 4.0 * n_boot as f64 { (self.bg.mean + z as f64 * self.bg.std()) as f32 } else { boot },
    }
}
// retrieve(): use self.thr() instead of self.cfg.sim_min; remember(): after storing, push 4 cosines
//   dot(normalized(emb), stored[(len * 2654435761 + j * 40503) >> 3 % len]) into self.bg   (needs read access to the index rows)
```
The pseudo-random pairs need row access to the index (`MemoryIndex` has none today): add `fn row(&self, i: usize) -> Option<Vec<f32>>`. (A ring of the last 64 queries would sample only recent records, which g10 did NOT measure; re-measure before using it.)
Cost: 4 dot products per stored query (negligible vs the flat scan).

## P2 (E2): recency discount of the per-anchor/agent vote statistics -- recommendation: opt-in only, default OFF

Measured: discounting speeds up recovery after a full context->agent remap on the F5 corpus (first 300 post-drift tasks +0.017..+0.027
success, recovery 120-240 tasks faster, replicated on MiniLM and BGE, held-out and confirm) but costs 0.01-0.025 H2000 in the stationary
world and nets H2000 +0.002..+0.006 (not Holm-significant) in the drift world; no effect on the G1 sibling corpus. `Event(0.98)` and
`Time(200)` leave the irrelevant-context control unchanged (|diff| <= 0.006); decaying the gate sums as well (`gate_disc`) breaks it (-0.035).

```rust
// knowledge.rs
pub struct VoteConfig { /* ... */ pub discount: Option<f64> }   // None = today (bit-identical). Some(g) = per-reward-event decay of the anchor row

// credit(): before `c.1 += r; c.2 += 1.0` for the anchor row:
if let (Some(g), Some(row)) = (self.cfg.vote.as_ref().and_then(|v| v.discount), self.astats.get_mut(&x.anchor)) {
    for c in row.iter_mut() { c.1 *= g; c.2 *= g; }          // reward sum and reward count of ALL agents of that anchor
}
```
Only `Event(gamma)` is proposed (O(agents) per reward, no clock needed); `Time(h)` measured the same within noise
(hl200 vs ev0.98: post-drift +0.017..+0.027 vs +0.015..+0.022 on F5). Recommended value if enabled: gamma = 0.98. Do NOT set `gate_disc`.
