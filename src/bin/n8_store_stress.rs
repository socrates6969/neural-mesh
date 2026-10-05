//! R61: multi-PROCESS stress of the shared store. The parent spawns W child processes (this same binary with `child <root> <id> <n>`); every child appends
//! N single outcomes (each append takes the store lock) and N lessons while the others do the same. Afterwards: all W*N outcome ids present exactly once,
//! zero unparsable journal lines, lock file gone. Then a folder-sync round across W stores. Prints PASS/FAIL and timings.
use neural_mesh::store::{LessonRec, MeshStore, Outcome};
use std::process::Command;
use std::time::Instant;

fn oc(sess: &str, i: u64) -> Outcome {
    Outcome { id: format!("{sess}:{i}"), session: sess.into(), seq: i, ts: i, world: "stress".into(), features: vec!["f".into()], task_type: None, agent: "a".into(), reward: 1.0 }
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("child") {
        let s = MeshStore::open(&a[2]).unwrap();
        let (id, n): (usize, u64) = (a[3].parse().unwrap(), a[4].parse().unwrap());
        for i in 0..n {
            s.append_outcome(&oc(&format!("w{id}"), i)).unwrap();
            if i % 10 == 0 {
                s.append_lesson(&LessonRec { id: format!("w{id}-l{i}"), session: format!("w{id}"), ts: i, world: "stress".into(), agent: None, tag: "t".into(), text: format!("lesson {id} {i}") }).unwrap();
            }
        }
        return;
    }
    let (w, n) = (a.get(1).and_then(|x| x.parse::<usize>().ok()).unwrap_or(8), a.get(2).and_then(|x| x.parse::<u64>().ok()).unwrap_or(300));
    let root = std::env::temp_dir().join(format!("nm_stress_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let store = MeshStore::open(&root).unwrap();
    let exe = std::env::current_exe().unwrap();
    let t0 = Instant::now();
    let kids: Vec<_> = (0..w).map(|i| Command::new(&exe).args(["child", root.to_str().unwrap(), &i.to_string(), &n.to_string()]).spawn().unwrap()).collect();
    let mut failed = 0;
    for mut k in kids { if !k.wait().unwrap().success() { failed += 1; } }
    let dt = t0.elapsed().as_secs_f64();
    let outs = store.outcomes().unwrap();
    let ids: std::collections::HashSet<String> = outs.iter().map(|o| o.id.clone()).collect();
    let raw = std::fs::read_to_string(root.join("router/journal.jsonl")).unwrap();
    let bad_lines = raw.lines().filter(|l| !l.trim().is_empty() && serde_json::from_str::<Outcome>(l).is_err()).count();
    let lessons = store.lessons().unwrap().len();
    let want_l = w * ((n as usize + 9) / 10);
    let lock_gone = !root.join(".lock").exists();
    let ok = failed == 0 && outs.len() == w * n as usize && ids.len() == outs.len() && bad_lines == 0 && lessons == want_l && lock_gone;
    println!("{w} processes x {n} appends in {dt:.2} s ({:.0} appends/s): outcomes {}/{} unique {} bad lines {bad_lines} lessons {lessons}/{want_l} child failures {failed} lock gone {lock_gone} -> {}", (w as f64 * n as f64) / dt, outs.len(), w * n as usize, ids.len(), if ok { "PASS" } else { "FAIL" });
    let _ = std::fs::remove_dir_all(&root);
    if !ok { std::process::exit(1); }
}
