//! Recommended DEFAULT router configuration (result of the E1 lab, `bin/e1_default`).
//!
//! `guarded(hebbian-robust)` with the gate re-tuned on a drift-heavy +
//! misleading-keyword family (tune seeds 1-8). Opt-in: nothing else in the crate
//! uses it; `HebbianConfig::default()` and `GuardConfig::default()` are unchanged.
//! Evidence: results/e1_default.json (held-out 31-60) and
//! results/e1_default_confirm61-90.json (frozen config, fresh seeds).

use std::collections::HashMap;

use crate::hebbian::{HebbianConfig, HebbianRouter};
use crate::prior::{GuardConfig, Guarded};

/// Gate: follow the keyword agent while its pooled Beta UCB >= 0.7 and no
/// challenger's LCB beats it; Beta(4,1) benefit of the doubt; counts discounted by
/// 0.98 per update of a feature row (trust can recover after drift).
pub fn recommended_guard() -> GuardConfig {
    GuardConfig { tau: 0.7, z: 0.5, kw_a: 4.0, kw_b: 1.0, discount: 0.98, challenge: true }
}

/// Inner learner after hand-over: "hebbian-robust" (eta .01, temperature .002,
/// decay .01, epsilon 0, other fields default).
pub fn recommended_inner() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

/// The recommended default router: keyword map in front, Hebbian learner behind.
pub fn recommended_router(keyword_map: HashMap<String, usize>, n_agents: usize) -> Guarded<HebbianRouter> {
    Guarded::new(HebbianRouter::new(n_agents, recommended_inner()), keyword_map, n_agents, recommended_guard())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::router::{Rng, Router, Task};

    #[test]
    fn recommended_follows_keywords_first_and_defaults_unchanged() {
        let mut map = HashMap::new();
        map.insert("sql".to_string(), 2usize);
        let mut r = recommended_router(map, 4);
        let t = Task { id: 0, task_type: None, features: vec!["sql".into()] };
        let mut rng = Rng::new(1);
        assert_eq!(r.select(&t, &[0, 1, 2, 3], &mut rng), 2);
        // the shared defaults are untouched
        assert_eq!(GuardConfig::default().tau, 0.6);
        assert_eq!(HebbianConfig::default().eta, 0.5);
    }
}
