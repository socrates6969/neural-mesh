//! R95: build a per-agent token ledger from recorded Claude Code transcripts (aggregates only; no content is kept or printed).
//! usage: n13_ledger [root=~/.claude/projects]   -> table per agent label + results/r95_ledger.json
use neural_mesh::ledger::{median_first_ctx, parse_transcript, Ledger, Pricing};
use std::path::{Path, PathBuf};
fn walk(p: &Path, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            let q = e.path();
            if q.is_dir() { walk(&q, out); } else if q.extension().map(|x| x == "jsonl").unwrap_or(false) { out.push(q); }
        }
    }
}
fn main() {
    let root = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap()).join(".claude").join("projects"));
    let mut files = vec![]; walk(&root, &mut files);
    let mut led = Ledger::new(); let t0 = std::time::Instant::now(); let mut bytes = 0u64;
    for f in &files {
        let Ok(text) = std::fs::read_to_string(f) else { continue }; bytes += text.len() as u64;
        let calls = parse_transcript(&text); if calls.is_empty() { continue; }
        let sub = f.components().any(|c| c.as_os_str() == "subagents");
        let label = if sub { std::fs::read_to_string(f.with_extension("meta.json")).ok().and_then(|m| serde_json::from_str::<serde_json::Value>(&m).ok()).and_then(|v| v.get("agentType").and_then(|x| x.as_str()).map(|s| s.to_string())).unwrap_or_else(|| "unknown".into()) } else { "main".to_string() };
        led.add_run(&label, &calls);
    }
    let p = Pricing::default(); let units = led.units(&p); let tu: f64 = units.values().sum();
    println!("{} files, {:.0} MB parsed in {:.1} s", files.len(), bytes as f64 / 1e6, t0.elapsed().as_secs_f64());
    let mut rows: Vec<_> = led.counted.iter().collect(); rows.sort_by(|a, b| units[b.0].partial_cmp(&units[a.0]).unwrap());
    println!("{:36} {:>5} {:>7} {:>10} {:>12} {:>10} {:>8}", "agent", "runs", "calls", "first_ctx", "tokens", "output", "units%");
    for (k, a) in rows.iter().take(12) { println!("{:36} {:5} {:7} {:10} {:12} {:10} {:8.1}", k, a.runs, a.calls, median_first_ctx(a).unwrap_or(0), a.usage.total(), a.usage.output, 100.0 * units[*k] / tu); }
    let t = led.total_counted(); println!("TOTAL counted tokens {} (input {} cache-write {} cache-read {} output {})", t.total(), t.input, t.cache_write, t.cache_read, t.output);
    std::fs::write("results/r95_ledger.json", led.to_json()).unwrap();
}
