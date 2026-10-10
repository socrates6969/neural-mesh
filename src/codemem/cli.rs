//! `neural-mesh codemem <query|update|status|hook|bench|eval|add|inject>` (also shipped as the small `codemem` binary: same code, faster process start).
//!   query "<q>" [--repo X | --cwd DIR] [--k 5] [--json]     file:line pointers with 1-2 snippet lines
//!   update [--repo X | --all] [--force]                     incremental (mtime+size, then hash); prints a JSON report
//!   status                                                  per-repo index age, files, chunks, bytes
//!   hook                                                    UserPromptSubmit hook: JSON on stdin, <=300-token block or nothing, exit 0
//!   inject --repo X "<q>"                                   what the hook would print for this prompt (for experiments), nothing if silent
//!   bench --repo X [--n 200] [--file Q.jsonl]               in-process query latency p50/p95/max
//!   eval --file Q.jsonl [--k 5]                             retrieval hit-rate / gate behaviour on a question set (see bench/codemem/)
//!   add NAME PATH                                           register a repo in data/codemem/config.json

use super::*;
use serde_json::{Value, json};

fn opt(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

fn flag(args: &[String], k: &str) -> bool {
    args.iter().any(|a| a == k)
}

/// positional args (not flags / flag values)
fn positional(args: &[String]) -> Vec<String> {
    const WITH_VAL: [&str; 8] = ["--repo", "--k", "--cwd", "--n", "--file", "--per-file", "--min-cov", "--split"];
    let mut out = Vec::new();
    let mut skip = false;
    for a in args {
        if skip {
            skip = false;
        } else if WITH_VAL.contains(&a.as_str()) {
            skip = true;
        } else if !a.starts_with("--") {
            out.push(a.clone());
        }
    }
    out
}

fn pick_repo<'a>(cfg: &'a Config, args: &[String]) -> Result<&'a RepoCfg, String> {
    if let Some(n) = opt(args, "--repo") {
        return cfg.repos.iter().find(|r| r.name == n).ok_or_else(|| format!("unknown repo {n}; known: {}", cfg.repos.iter().map(|r| r.name.as_str()).collect::<Vec<_>>().join(", ")));
    }
    let cwd = opt(args, "--cwd").or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().into_owned())).unwrap_or_default();
    repo_for_cwd(cfg, &cwd).ok_or_else(|| format!("cwd {cwd} is not inside a configured repo (use --repo)"))
}

fn print_pointers(repo: &str, ms: f64, s: &Search, ptrs: &[Pointer]) {
    println!("codemem repo={repo} hits={} coverage={:.2} {ms:.1}ms", ptrs.len(), s.coverage);
    for (i, p) in ptrs.iter().enumerate() {
        println!("{}. {}:{}  (chunk L{}-{}) [{}] score={:.1}", i + 1, p.path, p.line, p.l0, p.l1, p.title, p.score);
        for (l, t) in &p.snippet {
            println!("     L{l}: {t}");
        }
    }
}

fn cmd_query(cfg: &Config, dir: &Path, args: &[String]) -> Result<i32, String> {
    let repo = pick_repo(cfg, args)?;
    let q = positional(args).join(" ");
    if q.trim().is_empty() {
        return Err("usage: codemem query \"<question>\" [--repo X] [--k 5] [--json]".into());
    }
    let k: usize = opt(args, "--k").and_then(|v| v.parse().ok()).unwrap_or(5);
    let t0 = Instant::now();
    let mut ix = Index::open(&index_path(dir, &repo.name)).map_err(|e| format!("no index for {} ({e}); run: codemem update --repo {}", repo.name, repo.name))?;
    let spec = parse_query(&q, ix.split);
    let s = search(&ix, &spec, k, opt(args, "--per-file").and_then(|v| v.parse().ok()).unwrap_or(2));
    let ptrs: Vec<Pointer> = s.hits.iter().map(|h| pointer(&mut ix, h, &spec, 110)).collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    if flag(args, "--json") {
        println!("{}", json!({"repo": repo.name, "ms": ms, "coverage": s.coverage, "pointers": ptrs}));
    } else {
        print_pointers(&repo.name, ms, &s, &ptrs);
    }
    Ok(0)
}

fn cmd_update(cfg: &Config, dir: &Path, args: &[String]) -> Result<i32, String> {
    let force = flag(args, "--force");
    let repos: Vec<&RepoCfg> = if flag(args, "--all") || opt(args, "--repo").is_none() && opt(args, "--cwd").is_none() { cfg.repos.iter().collect() } else { vec![pick_repo(cfg, args)?] };
    let mut worst = 0;
    for r in repos {
        match update_repo(cfg, r, dir, force) {
            Ok(rep) => println!("{}", serde_json::to_string(&rep).unwrap_or_default()),
            Err(e) => {
                println!("{}", json!({"repo": r.name, "error": e}));
                worst = 1;
            }
        }
    }
    Ok(worst)
}

fn cmd_status(cfg: &Config, dir: &Path) -> Result<i32, String> {
    for r in &cfg.repos {
        let p = index_path(dir, &r.name);
        let v = match Index::open(&p) {
            Ok(ix) => json!({"repo": r.name, "files": ix.n_files, "chunks": ix.n_chunks, "terms": ix.n_terms, "bytes": std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), "age_s": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs().saturating_sub(ix.created)).unwrap_or(0)}),
            Err(_) => json!({"repo": r.name, "indexed": false}),
        };
        println!("{v}");
    }
    Ok(0)
}

fn pctl(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(((v.len() as f64 * p).ceil() as usize).saturating_sub(1).min(v.len().saturating_sub(1))).copied().unwrap_or(0.0)
}

fn cmd_bench(cfg: &Config, dir: &Path, args: &[String]) -> Result<i32, String> {
    let repo = pick_repo(cfg, args)?;
    let n: usize = opt(args, "--n").and_then(|v| v.parse().ok()).unwrap_or(200);
    let qs: Vec<String> = match opt(args, "--file") {
        Some(f) => read_questions(&f)?.into_iter().map(|q| q.question).collect(),
        None => vec!["where is tenant isolation enforced".into(), "which file defines withTenant".into(), "`set_config` app.tenant_id".into(), "BM25 k1 b retrieval profile".into(), "migration rls policy force row level security".into()],
    };
    let mut lat = Vec::new();
    for i in 0..n {
        let t0 = Instant::now();
        let mut ix = Index::open(&index_path(dir, &repo.name)).map_err(|e| e.to_string())?;
        let spec = parse_query(&qs[i % qs.len()], ix.split);
        let s = search(&ix, &spec, 5, 2);
        let _: Vec<Pointer> = s.hits.iter().map(|h| pointer(&mut ix, h, &spec, 110)).collect();
        lat.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let (p50, p95, mx) = (pctl(&mut lat.clone(), 0.5), pctl(&mut lat.clone(), 0.95), pctl(&mut lat, 1.0));
    println!("{}", json!({"repo": repo.name, "n": n, "includes": "open+load index, parse, search, 5 pointers", "p50_ms": p50, "p95_ms": p95, "max_ms": mx}));
    Ok(0)
}

pub struct Question {
    pub id: String,
    pub question: String,
    pub repo: String,
    pub answers: Vec<(String, u32, u32)>,
    pub split: String,
}

pub fn read_questions(path: &str) -> Result<Vec<Question>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for l in s.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(l).map_err(|e| format!("bad jsonl line: {e}"))?;
        let answers = v["answers"].as_array().map(|a| a.iter().map(|x| (x["file"].as_str().unwrap_or("").to_string(), x["l0"].as_u64().unwrap_or(0) as u32, x["l1"].as_u64().unwrap_or(0) as u32)).collect()).unwrap_or_default();
        out.push(Question { split: v["split"].as_str().unwrap_or("").into(), id: v["id"].as_str().unwrap_or("").into(), question: v["question"].as_str().unwrap_or("").into(), repo: v["repo"].as_str().unwrap_or("").into(), answers });
    }
    Ok(out)
}

/// does a pointer satisfy an answer key entry? file must match; if the key has a line range, the pointer's chunk must overlap it (+-3 lines).
fn matches_key(p: &Pointer, ans: &[(String, u32, u32)], need_line: bool) -> bool {
    ans.iter().any(|(f, a, b)| p.path == *f && (!need_line || *a == 0 || (p.l0 <= *b + 3 && p.l1 + 3 >= *a)))
}

fn cmd_eval(cfg: &Config, dir: &Path, args: &[String]) -> Result<i32, String> {
    let mut qs = read_questions(&opt(args, "--file").ok_or("--file required")?)?;
    if let Some(sp) = opt(args, "--split") {
        qs.retain(|q| q.split == sp);
    }
    let k: usize = opt(args, "--k").and_then(|v| v.parse().ok()).unwrap_or(5);
    let mut rows = Vec::new();
    for q in &qs {
        let repo = cfg.repos.iter().find(|r| r.name == q.repo).ok_or_else(|| format!("{}: unknown repo {}", q.id, q.repo))?;
        let mut ix = Index::open(&index_path(dir, &repo.name)).map_err(|e| e.to_string())?;
        let spec = parse_query(&q.question, ix.split);
        let s = search(&ix, &spec, k, 2);
        let ptrs: Vec<Pointer> = s.hits.iter().map(|h| pointer(&mut ix, h, &spec, 110)).collect();
        let rank = |line: bool| ptrs.iter().position(|p| matches_key(p, &q.answers, line)).map(|r| r + 1);
        let gate = gate::evaluate(&q.question, &repo.terms, &repo.name).fire;
        let floor = s.coverage >= gate::MIN_COVERAGE && s.rare_matched >= gate::MIN_RARE && s.matched >= gate::MIN_MATCHED;
        rows.push(json!({"id": q.id, "repo": q.repo, "file_rank": rank(false), "line_rank": rank(true), "coverage": s.coverage, "rare": s.rare_matched, "matched": s.matched, "gate": gate, "floor": floor, "fires": gate && floor}));
    }
    let n = rows.len() as f64;
    let hit = |key: &str, kk: u64| rows.iter().filter(|r| r[key].as_u64().is_some_and(|x| x <= kk)).count() as f64 / n;
    let fired: Vec<&Value> = rows.iter().filter(|r| r["fires"] == true).collect();
    let fired_right = fired.iter().filter(|r| r["file_rank"].as_u64().is_some()).count();
    println!("{}", json!({"n": rows.len(), "file_hit@1": hit("file_rank", 1), "file_hit@3": hit("file_rank", 3), "file_hit@5": hit("file_rank", 5), "line_hit@1": hit("line_rank", 1), "line_hit@3": hit("line_rank", 3), "line_hit@5": hit("line_rank", 5),
        "fires": fired.len(), "fires_with_right_file_in_topk": fired_right, "rows": if flag(args, "--rows") { Value::Array(rows.clone()) } else { Value::Null }}));
    Ok(0)
}

fn cmd_add(dir: &Path, args: &[String]) -> Result<i32, String> {
    let pos = positional(args);
    let (name, path) = (pos.first().ok_or("usage: add NAME PATH")?, pos.get(1).ok_or("usage: add NAME PATH")?);
    let mut cfg = load_config(dir);
    cfg.repos.retain(|r| r.name != *name);
    cfg.repos.push(RepoCfg { name: name.clone(), path: path.replace('\\', "/"), exclude: vec![], extra_roots: vec![], terms: vec![] });
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap_or_default()).map_err(|e| e.to_string())?;
    println!("{}", json!({"ok": true, "repos": cfg.repos.len()}));
    Ok(0)
}

fn cmd_inject(dir: &Path, args: &[String]) -> Result<i32, String> {
    let cfg = load_config(dir);
    let repo = pick_repo(&cfg, args)?;
    let input = json!({"prompt": positional(args).join(" "), "cwd": repo.path}).to_string();
    if let Some(t) = gate::decide(&input, dir).text {
        print!("{t}");
    }
    Ok(0)
}

/// CLI entry. `hook` never reports errors (silent, exit 0); everything else prints errors to stderr and exits 1.
pub fn run(args: &[String]) -> i32 {
    let sub = args.first().map(String::as_str).unwrap_or("");
    if sub == "hook" {
        return gate::run_hook();
    }
    let dir = data_dir();
    let cfg = load_config(&dir);
    let rest = &args[args.len().min(1)..];
    let r = match sub {
        "query" => cmd_query(&cfg, &dir, rest),
        "update" => cmd_update(&cfg, &dir, rest),
        "status" => cmd_status(&cfg, &dir),
        "bench" => cmd_bench(&cfg, &dir, rest),
        "eval" => cmd_eval(&cfg, &dir, rest),
        "add" => cmd_add(&dir, rest),
        "inject" => cmd_inject(&dir, rest),
        "gate" => {
            let g = gate::evaluate(&positional(rest).join(" "), &[], "");
            println!("{}", json!({"score": g.score, "fire": g.fire, "reason": g.reason}));
            Ok(0)
        }
        _ => Err("usage: codemem <query|update|status|hook|inject|bench|eval|add> ...".into()),
    };
    r.unwrap_or_else(|e| {
        eprintln!("codemem: {e}");
        1
    })
}
