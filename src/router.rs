//! Shared router contract. See DESIGN.md. Do not change the trait signature.

/// Small seeded PRNG (mulberry32-style) so every component is deterministic.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15)) }
    pub fn next_u64(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0,1).
    pub fn f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
    pub fn below(&mut self, n: usize) -> usize { (self.f64() * n as f64) as usize % n.max(1) }
    pub fn bernoulli(&mut self, p: f64) -> bool { self.f64() < p }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Task {
    pub id: u64,
    pub task_type: Option<String>,
    pub features: Vec<String>,
}

pub trait Router {
    fn name(&self) -> String;
    /// Pick one agent from `candidates` (indices into the agent list, all eligible).
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize;
    /// Outcome feedback for the agent that was chosen. `reward` in [0,1].
    fn update(&mut self, task: &Task, agent: usize, reward: f64);
}
