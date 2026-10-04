//! Export a real trained brain for the dashboard: the accepted router
//! (Warm keyword prior + robust Hebbian, inside a team_size=1 NeuralMesh for
//! health benching) trained on the fixed benchmark world, held-out seed 31.
//! Writes results/brain.json. Read-only use of the library.
use neural_mesh::baselines::KeywordRouter;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{World, WorldConfig, AGENTS, TYPES};
use serde_json::json;

fn main() {
    let seed = 31u64;
    let world = World::new(seed, WorldConfig::default());
    let n = world.n_agents();
    let cfg = HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() };
    let warm = Warm::new(HebbianRouter::new(n, cfg), World::keyword_map(), n, 1.0).with_evidence(Evidence::MeanFeature);
    let mut mesh = NeuralMesh::new(n, warm, MeshConfig { team_size: 1, ..MeshConfig::default() });
    let mut kw = KeywordRouter::new(World::keyword_map(), n);
    let mut rng = Rng::new(seed * 1_000_003 + 17);
    let mut krng = Rng::new(seed * 1_000_003 + 17);
    let cands: Vec<usize> = (0..n).collect();
    let (mut s_brain, mut s_kw, mut s_or, mut kwfollow) = (0.0, 0.0, 0.0, 0.0);
    let mut timeline = Vec::new();
    let mut health_tl = Vec::new();
    let window = 100usize;
    for t in 0..world.len() {
        let st = world.task(t);
        let kw_w = mesh.leader().keyword_weight(&st.task);
        let team = mesh.route(&st.task, &cands, &mut rng);
        let r = world.outcome(t, &team[..1]);
        mesh.feedback(&st.task, &team[..1], r);
        let ka = kw.select(&st.task, &cands, &mut krng);
        s_brain += world.success_prob(t, &team[..1]);
        s_kw += world.success_prob(t, &[ka]);
        s_or += (0..n).map(|a| world.success_prob(t, &[a])).fold(0.0, f64::max); // best available single agent
        kwfollow += kw_w;
        if (t + 1) % window == 0 {
            let w = window as f64;
            timeline.push(json!({"t": t + 1, "brain": s_brain / w, "keyword": s_kw / w, "oracle": s_or / w,
                                 "keyword_reliance": kwfollow / w, "in_failure": world.in_failure(t)}));
            health_tl.push(json!({"t": t + 1, "health": (0..n).map(|a| mesh.health(a)).collect::<Vec<_>>()}));
            s_brain = 0.0; s_kw = 0.0; s_or = 0.0; kwfollow = 0.0;
        }
    }
    let heb = &mesh.leader().inner;
    let mut rows = Vec::new();
    for (feat, ws) in heb.rows() {
        rows.push(json!({"feature": feat, "weights": ws, "baseline": heb.baseline(feat)}));
    }
    let out = json!({
        "router": "warm(hebbian-robust) k=1 MeanFeature + health benching (team_size 1)",
        "world": "fixed benchmark world, held-out seed 31, 2000 tasks, drift at 1000",
        "agents": AGENTS, "types": TYPES, "drift_at": world.drift_at(),
        "timeline": timeline, "health": health_tl, "weights": rows,
        "final_health": (0..n).map(|a| mesh.health(a)).collect::<Vec<_>>(),
    });
    std::fs::create_dir_all("results").unwrap();
    std::fs::write("results/brain.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote results/brain.json: {} features, {} timeline points", rows.len(), timeline.len());
}
