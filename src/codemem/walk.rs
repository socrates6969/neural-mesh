//! File discovery and the exclusion policy. Git repos are listed by `git ls-files -z -co --exclude-standard` (tracked + untracked-not-ignored,
//! i.e. .gitignore is honoured by git itself); other directories are walked. Then the allow-list below decides what is read at all.

use super::RepoCfg;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const MAX_BYTES: u64 = 400_000;
const MAX_JSON_BYTES: u64 = 200_000;

const SKIP_DIRS: [&str; 24] = [
    "node_modules", "target", "dist", "build", ".git", ".next", "__pycache__", ".venv", "venv", "coverage", ".terraform", ".cache", "out",
    ".pytest_cache", "playwright-report", "test-results", "secrets", "secret", "data", ".swarm", ".claude-flow", ".idea", ".vscode", "vendor",
];

const ALLOWED_EXT: [&str; 31] = [
    "rs", "py", "ts", "tsx", "js", "jsx", "mjs", "cjs", "md", "mdx", "txt", "toml", "yaml", "yml", "json", "sql", "prisma", "tf", "ps1", "sh",
    "bash", "css", "scss", "html", "cfg", "ini", "conf", "bat", "cmd", "graphql", "proto",
];

const ALLOWED_NAMES: [&str; 4] = ["dockerfile", "makefile", "codeowners", "justfile"];

fn glob_hit(rel: &str, pat: &str) -> bool {
    match pat.strip_suffix('*') {
        Some(p) => rel.starts_with(p),
        None => rel == pat || rel.starts_with(pat),
    }
}

/// true when `rel` (a `/`-separated path relative to the repo, or an absolute extra-root path) may be indexed.
pub fn should_index(rel: &str, size: u64, exclude: &[String]) -> bool {
    let lower = rel.to_ascii_lowercase();
    let parts: Vec<&str> = lower.split('/').filter(|p| !p.is_empty()).collect();
    let Some(name) = parts.last().copied() else { return false };
    let dirs = &parts[..parts.len() - 1];
    let skip_dir = |d: &&str| SKIP_DIRS.contains(d) || d.starts_with("vendor_") || (d.starts_with('.') && !matches!(*d, ".github" | ".claude") && d.len() > 1 && !d.ends_with(':'));
    if dirs.iter().filter(|d| !d.ends_with(':')).any(|d| skip_dir(d)) {
        return false;
    }
    if name.starts_with(".env") || name.ends_with(".key") || name.ends_with(".pem") || name.starts_with("id_rsa") || name.starts_with("id_ed25519") {
        return false;
    }
    let sensitive = ["credential", "service-account", "serviceaccount", "client_secret", ".tfvars", ".tfstate", ".p12", ".pfx", ".jks", ".kdbx", "settings.local.json", ".npmrc", ".netrc", ".pgpass"];
    if sensitive.iter().any(|s| name.contains(s)) || matches!(name, "token.json" | "tokens.json" | "auth.json") {
        return false;
    }
    if name.ends_with(".lock") || name.ends_with("-lock.json") || name.ends_with(".min.js") || name.ends_with(".map") || name.ends_with(".d.ts") {
        return false;
    }
    let ext = name.rsplit_once('.').map(|x| x.1).unwrap_or("");
    if !(ALLOWED_EXT.contains(&ext) || ALLOWED_NAMES.contains(&name)) {
        return false;
    }
    let cap = if ext == "json" { MAX_JSON_BYTES } else { MAX_BYTES };
    size <= cap && !exclude.iter().any(|p| glob_hit(rel, p))
}

pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(8192).any(|b| *b == 0)
}

fn git_files(root: &Path) -> Option<Vec<String>> {
    let out = Command::new("git").arg("-C").arg(root).args(["ls-files", "-z", "-co", "--exclude-standard"]).stdin(Stdio::null()).stderr(Stdio::null()).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(out.stdout.split(|b| *b == 0).filter(|s| !s.is_empty()).map(|s| String::from_utf8_lossy(s).into_owned()).collect())
}

fn walk_dir(root: &Path, dir: &Path, out: &mut Vec<String>, depth: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let (p, name) = (e.path(), e.file_name().to_string_lossy().to_ascii_lowercase());
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() && depth < 12 && !SKIP_DIRS.contains(&name.as_str()) && !(name.starts_with('.') && name != ".github" && name != ".claude") {
            walk_dir(root, &p, out, depth + 1);
        } else if ft.is_file() {
            if let Ok(r) = p.strip_prefix(root) {
                out.push(r.to_string_lossy().replace('\\', "/"));
            }
        }
    }
}

/// (display path, absolute path) of every indexable file, sorted by display path.
pub fn list_files(repo: &RepoCfg) -> Result<Vec<(String, PathBuf)>, String> {
    let root = PathBuf::from(&repo.path);
    if !root.is_dir() {
        return Err(format!("repo path not found: {}", repo.path));
    }
    let rels = match root.join(".git").exists().then(|| git_files(&root)).flatten() {
        Some(v) => v,
        None => {
            let mut v = Vec::new();
            walk_dir(&root, &root, &mut v, 0);
            v
        }
    };
    // (display, absolute, path used for the policy, excludes apply)
    let mut all: Vec<(String, PathBuf, String, bool)> = rels.into_iter().map(|r| (r.clone(), root.join(&r), r, true)).collect();
    for extra in &repo.extra_roots {
        let (er, mut v) = (PathBuf::from(extra), Vec::new());
        walk_dir(&er, &er, &mut v, 0);
        all.extend(v.into_iter().map(|r| (format!("{}/{}", extra.trim_end_matches('/'), r), er.join(&r), r, false)));
    }
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    for (disp, abs, policy, own) in all {
        let size = std::fs::metadata(&abs).map(|m| if m.is_file() { m.len() } else { u64::MAX }).unwrap_or(u64::MAX);
        if should_index(&policy, size, if own { &repo.exclude } else { &[] }) {
            out.push((disp, abs));
        }
    }
    out.sort();
    out.dedup_by(|a, b| a.0 == b.0);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_and_deny() {
        let ok = |p: &str| should_index(p, 100, &[]);
        assert!(ok("api/src/db/tenant.ts") && ok("docs/SPEC.md") && ok(".github/workflows/web-ci.yml") && ok(".claude/agents/x.md") && ok("Dockerfile"));
        for bad in ["secrets/a.md", ".env", ".env.example", "api/.env.local", "certs/server.key", "a/b.pem", "data/x.json", "node_modules/a/b.js", "package-lock.json", "Cargo.lock",
            "a/logo.png", "target/release/x.rs", ".swarm/memory.json", "vendor_g17/a.rs", "infra/prod.tfvars", ".claude/settings.local.json", "a/credentials.json", "web/dist/app.js"] {
            assert!(!ok(bad), "{bad} must be excluded");
        }
        assert!(!should_index("big.rs", 500_000, &[]));
        assert!(!should_index("bench/battery.json", 10, &["bench/battery".to_string()]));
    }
}
