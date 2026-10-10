//! UserPromptSubmit hook logic: cheap lexical gate -> repo from cwd -> BM25 -> relevance floor -> a <= 300-token block, or nothing.
//! Never panics out, never writes to stdout on any problem (see `run_hook`).

use super::*;
use std::io::Read;

/// Output cap. Pointer lines are code, ~3 chars/token worst case: 900 chars ~ 300 tokens.
pub const MAX_CHARS: usize = 900;
pub const MAX_POINTERS: usize = 5;
const LINE_CHARS: usize = 78;
/// Relevance floor (tuned on the TUNE questions, bench/codemem/README.md).
pub const MIN_COVERAGE: f32 = 0.20;
pub const MIN_RARE: usize = 1;
pub const MIN_MATCHED: usize = 2;
const MIN_GATE: u32 = 3;

const CODE_NOUNS: [&str; 24] = [
    "function", "file", "class", "module", "endpoint", "migration", "schema", "test", "tests", "config", "handler", "middleware", "route", "table",
    "policy", "hook", "script", "enforced", "defined", "implemented", "constant", "repo", "codebase", "variable",
];
/// word stems of "where is X <done>" questions (prefix match, `set` exact)
const Q_STEMS: [&str; 24] = [
    "defin", "enforc", "implement", "check", "validat", "reject", "creat", "declar", "install", "comput", "call", "configur", "stor", "document",
    "produc", "scrub", "hash", "resolv", "perform", "handl", "decid", "block", "limit", "mint",
];
const Q_WORDS: [&str; 8] = ["where", "which", "what", "how", "who", "why", "find", "show"];

#[derive(Debug, PartialEq)]
pub struct Gate {
    pub score: u32,
    pub fire: bool,
    pub reason: &'static str,
}

pub fn is_ident_like(w: &str) -> bool {
    let w = w.trim_matches(|c: char| !(c.is_alphanumeric() || c == '_'));
    let has_us = w.contains('_') && w.len() > 3 && w.chars().any(|c| c.is_alphabetic()) && !w.starts_with('_') && !w.ends_with('_');
    let inner_cap = w.len() > 3 && w.chars().skip(1).any(|c| c.is_uppercase()) && w.chars().any(|c| c.is_lowercase()) && w.chars().all(|c| c.is_alphanumeric());
    has_us || inner_cap
}

fn path_like(w: &str) -> bool {
    let w = w.trim_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | '\\')));
    is_path_word(w)
}

pub fn evaluate(prompt: &str, repo_terms: &[String], repo_name: &str) -> Gate {
    let t = prompt.trim();
    let words: Vec<&str> = t.split_whitespace().collect();
    let slash_cmd = words.first().is_some_and(|w| w.starts_with('/') && w[1..].chars().all(|c| c.is_alphanumeric() || matches!(c, ':' | '-' | '_')));
    if slash_cmd || words.len() < 3 && !words.iter().any(|w| path_like(w)) {
        return Gate { score: 0, fire: false, reason: "short" };
    }
    let lower = t.to_ascii_lowercase();
    let mut score = 0u32;
    score += 3 * words.iter().filter(|w| path_like(w)).count().min(1) as u32;
    score += 2 * (t.matches('`').count() / 2).min(2) as u32;
    score += 2 * words.iter().filter(|w| is_ident_like(w)).count().min(2) as u32;
    score += (t.contains("::") || t.contains("()") || t.contains(" --")) as u32 * 2;
    let lw: Vec<String> = lower.split(|c: char| !c.is_alphanumeric() && c != '_').map(str::to_string).collect();
    let nouns = CODE_NOUNS.iter().filter(|n| lw.iter().any(|w| w == **n)).count().min(2) as u32;
    score += nouns;
    let q_form = lw.first().is_some_and(|w| Q_WORDS.contains(&w.as_str()));
    if q_form && nouns > 0 {
        score += 2;
    }
    if q_form && lw.iter().any(|w| w == "set" || w == "used" || Q_STEMS.iter().any(|st| w.starts_with(st))) {
        score += 3;
    }
    if lower.contains(&repo_name.to_ascii_lowercase()) || repo_terms.iter().any(|x| lw.iter().any(|w| *w == x.to_ascii_lowercase())) {
        score += 1;
    }
    let fire = score >= MIN_GATE;
    Gate { score, fire, reason: if fire { "gate" } else { "weak" } }
}

/// First 700 chars of the prompt plus every identifier / path-like word beyond them (long pasted prompts stay cheap and on topic).
pub fn query_text(prompt: &str) -> String {
    let head: String = prompt.chars().take(700).collect();
    let rest = prompt.get(head.len()..).unwrap_or("");
    let extra: Vec<&str> = rest.split_whitespace().filter(|w| is_ident_like(w) || path_like(w)).take(30).collect();
    format!("{head} {}", extra.join(" "))
}

/// Renders pointers under the char cap; returns None when nothing fits.
pub fn render(repo: &str, ptrs: &[Pointer]) -> Option<String> {
    let head = format!("[codemem] BM25 pointers in {repo} (heuristic; verify with Read):");
    let mut out = head.clone();
    let mut n = 0;
    for p in ptrs.iter().take(MAX_POINTERS) {
        let sn: Vec<String> = p.snippet.iter().take(2).map(|(l, s)| format!("L{l} {s}")).collect();
        let title = if p.title == "(top)" || p.title == "(window)" { String::new() } else { format!(" [{}]", p.title) };
        let line = format!("\n- {}:{}{} {}", p.path, p.line, title, sn.join(" | "));
        let line = if line.chars().count() > 230 { format!("{}…", line.chars().take(229).collect::<String>()) } else { line };
        if out.chars().count() + line.chars().count() > MAX_CHARS {
            break;
        }
        out.push_str(&line);
        n += 1;
    }
    (n > 0).then_some(out)
}

pub struct HookOutcome {
    pub text: Option<String>,
    pub reason: String,
    pub repo: String,
    pub n: usize,
}

fn quiet(reason: &str) -> HookOutcome {
    HookOutcome { text: None, reason: reason.into(), repo: String::new(), n: 0 }
}

/// The whole decision, from the hook JSON. Pure of stdout.
pub fn decide(input: &str, dir: &Path) -> HookOutcome {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(input) else { return quiet("bad_json") };
    let (prompt, cwd) = (v["prompt"].as_str().unwrap_or(""), v["cwd"].as_str().unwrap_or(""));
    let cfg = load_config(dir);
    let Some(repo) = repo_for_cwd(&cfg, cwd) else { return quiet("no_repo") };
    let g = evaluate(prompt, &repo.terms, &repo.name);
    if !g.fire {
        return HookOutcome { repo: repo.name.clone(), ..quiet(g.reason) };
    }
    let Ok(mut ix) = Index::open(&index_path(dir, &repo.name)) else { return HookOutcome { repo: repo.name.clone(), ..quiet("no_index") } };
    let q = parse_query(&query_text(prompt), ix.split);
    let s = search(&ix, &q, MAX_POINTERS, 2);
    if s.coverage < MIN_COVERAGE || s.rare_matched < MIN_RARE || s.matched < MIN_MATCHED {
        return HookOutcome { repo: repo.name.clone(), ..quiet("below_floor") };
    }
    let ptrs: Vec<Pointer> = s.hits.iter().map(|h| pointer(&mut ix, h, &q, LINE_CHARS)).collect();
    let text = render(&repo.name, &ptrs);
    HookOutcome { n: ptrs.len(), reason: "fired".into(), repo: repo.name.clone(), text }
}

fn log_line(dir: &Path, o: &HookOutcome, ms: u128, session: &str) {
    use std::io::Write;
    if std::env::var("NEURAL_MESH_CODEMEM_LOG").map(|v| v.eq_ignore_ascii_case("off")).unwrap_or(false) {
        return;
    }
    let line = serde_json::json!({"ts": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "session": session, "repo": o.repo, "reason": o.reason, "fired": o.text.is_some(), "chars": o.text.as_ref().map(|t| t.chars().count()).unwrap_or(0), "ms": ms});
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("hook.jsonl")) {
        let _ = writeln!(f, "{line}");
    }
}

/// Entry point of `codemem hook`: stdin JSON -> stdout block or nothing. Always returns exit code 0.
pub fn run_hook() -> i32 {
    if std::env::var("NEURAL_MESH_CODEMEM").map(|v| v.trim().eq_ignore_ascii_case("off")).unwrap_or(false) {
        return 0;
    }
    let budget: u64 = std::env::var("NEURAL_MESH_CODEMEM_BUDGET_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let mut input = String::new();
    if std::io::stdin().take(1 << 20).read_to_string(&mut input).is_err() {
        return 0;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let t0 = Instant::now();
    let inp = input.clone();
    std::thread::spawn(move || {
        let dir = data_dir();
        let r = std::panic::catch_unwind(|| decide(&inp, &dir));
        let _ = tx.send((r, dir));
    });
    if let Ok((Ok(o), dir)) = rx.recv_timeout(std::time::Duration::from_millis(budget)) {
        let session = serde_json::from_str::<serde_json::Value>(&input).ok().and_then(|v| v["session_id"].as_str().map(str::to_string)).unwrap_or_default();
        log_line(&dir, &o, t0.elapsed().as_millis(), &session);
        if let Some(t) = o.text {
            print!("{t}");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms() -> Vec<String> {
        vec!["tenant".into()]
    }

    #[test]
    fn gate_fires_on_code_signals_and_stays_silent_on_chatter() {
        for p in [
            "where is tenant isolation enforced in api/src/db/tenant.ts",
            "what does `withTenant` do",
            "which file defines the set_config call",
            "How does the migration for the audit chain work in this repo?",
        ] {
            assert!(evaluate(p, &terms(), "x").fire, "{p}");
        }
        for p in ["continue", "yes please do it", "/loop 5m", "thanks, looks good to me", "can you make it shorter and nicer"] {
            assert!(!evaluate(p, &terms(), "x").fire, "{p}");
        }
    }

    #[test]
    fn render_respects_char_cap_and_five_pointers() {
        let p = |i: u32| Pointer { path: format!("some/long/path/to/file_{i}.ts"), line: i, l0: i, l1: i + 9, title: "function withSomethingLong".into(), score: 1.0,
            snippet: vec![(i, "x".repeat(200)), (i + 1, "y".repeat(200))] };
        let ptrs: Vec<Pointer> = (1..=8).map(p).collect();
        let t = render("repo", &ptrs).unwrap();
        assert!(t.chars().count() <= MAX_CHARS, "{}", t.chars().count());
        assert!(t.matches("\n- ").count() <= MAX_POINTERS);
        assert!(render("repo", &[]).is_none());
    }

    fn fixture() -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!("cm-hook-root-{}", std::process::id()));
        let data = std::env::temp_dir().join(format!("cm-hook-data-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&data);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/tenant.ts"), "export async function withTenant(id: string) {\n  await tx.$executeRaw`select set_config('app.tenant_id', ${id}, true)`;\n}\n").unwrap();
        std::fs::write(root.join("src/a.ts"), "export function filler() { return 1; }\n").unwrap();
        let cfg = Config { split: false, repos: vec![RepoCfg { name: "t".into(), path: root.to_string_lossy().replace('\\', "/"), exclude: vec![], extra_roots: vec![], terms: terms() }] };
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("config.json"), serde_json::to_string(&cfg).unwrap()).unwrap();
        update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        (root, data)
    }

    #[test]
    fn decide_fires_for_in_repo_code_question_and_is_silent_otherwise() {
        let (root, data) = fixture();
        let cwd = root.to_string_lossy().replace('\\', "/");
        let j = |p: &str, c: &str| serde_json::json!({"prompt": p, "cwd": c, "session_id": "s"}).to_string();
        let o = decide(&j("where is `withTenant` and set_config defined in src/tenant.ts", &cwd), &data);
        let t = o.text.expect("should fire");
        assert!(t.contains("src/tenant.ts:2") || t.contains("src/tenant.ts:1"), "{t}");
        assert!(decide(&j("where is `withTenant` and set_config defined", "C:/elsewhere"), &data).text.is_none());
        assert!(decide(&j("continue", &cwd), &data).text.is_none());
        assert!(decide(&j("where is `zzqqxx_unknown_ident` defined in the repo", &cwd), &data).text.is_none(), "below the relevance floor");
        assert!(decide("not json", &data).text.is_none());
        assert!(decide("{}", &data).text.is_none());
        let empty = std::env::temp_dir().join("cm-nonexistent-dir-xyz");
        assert!(decide(&j("where is `withTenant` defined in src/tenant.ts", &cwd), &empty).text.is_none(), "missing config/index is silent");
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(data);
    }
}
