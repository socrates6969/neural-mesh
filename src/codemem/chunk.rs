//! Chunking by symbol (code) or heading (docs), each chunk carrying its 1-based inclusive line range. Light-weight line heuristics, no parser:
//! a boundary is a top-level-ish declaration (indent <= 4) preceded by its doc comment / attributes. Segments longer than MAX_LINES are cut.

pub const MAX_LINES: usize = 70;
pub const MAX_BYTES: usize = 5000;

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkDraft {
    pub l0: u32,
    pub l1: u32,
    pub title: String,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Lang {
    Rust,
    Py,
    Ts,
    Sql,
    Prisma,
    Tf,
    Shell,
    Md,
    Other,
}

fn lang_of(path: &str) -> Lang {
    let ext = path.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Lang::Rust,
        "py" => Lang::Py,
        "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" => Lang::Ts,
        "sql" => Lang::Sql,
        "prisma" => Lang::Prisma,
        "tf" => Lang::Tf,
        "ps1" | "sh" | "bash" => Lang::Shell,
        "md" | "mdx" | "txt" => Lang::Md,
        _ => Lang::Other,
    }
}

fn indent_of(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ' || *c == '\t').map(|c| if c == '\t' { 4 } else { 1 }).sum()
}

fn ident_after(s: &str) -> String {
    s.trim_start().chars().take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$').collect()
}

fn rust_symbol(t: &str) -> Option<String> {
    let mut s = t;
    if let Some(r) = s.strip_prefix("pub") {
        s = if r.starts_with('(') { r.split_once(')').map(|x| x.1).unwrap_or("") } else { r };
        s = s.trim_start();
    }
    for w in ["async ", "unsafe ", "const ", "extern \"C\" "] {
        if let Some(r) = s.strip_prefix(w) {
            s = r;
        }
    }
    for kw in ["fn ", "struct ", "enum ", "trait ", "mod ", "type ", "static ", "union ", "const "] {
        if let Some(r) = s.strip_prefix(kw) {
            let n = ident_after(r);
            return if n.is_empty() { None } else { Some(format!("{}{}", kw, n)) };
        }
    }
    if s.starts_with("impl") && (s[4..].starts_with(' ') || s[4..].starts_with('<')) {
        return Some(s.split('{').next().unwrap_or(s).trim().chars().take(70).collect());
    }
    s.strip_prefix("macro_rules! ").map(|r| format!("macro_rules! {}", ident_after(r)))
}

fn py_symbol(t: &str) -> Option<String> {
    let s = t.strip_prefix("async ").unwrap_or(t);
    for kw in ["def ", "class "] {
        if let Some(r) = s.strip_prefix(kw) {
            return Some(format!("{}{}", kw, ident_after(r)));
        }
    }
    None
}

fn ts_symbol(t: &str, indent: usize) -> Option<String> {
    let mut s = t;
    for w in ["export ", "default ", "declare ", "async ", "abstract "] {
        if let Some(r) = s.strip_prefix(w) {
            s = r;
        }
    }
    let s = s.strip_prefix("async ").unwrap_or(s);
    for kw in ["function* ", "function ", "class ", "interface ", "type ", "enum ", "namespace "] {
        if let Some(r) = s.strip_prefix(kw) {
            let n = ident_after(r);
            if !n.is_empty() {
                return Some(format!("{}{}", kw.trim_end_matches('*').trim_end(), format_args!(" {}", n)));
            }
        }
    }
    if indent == 0 {
        for kw in ["const ", "let ", "var "] {
            if let Some(r) = s.strip_prefix(kw) {
                let n = ident_after(r);
                if !n.is_empty() {
                    return Some(format!("{}{}", kw, n));
                }
            }
        }
    }
    for kw in ["describe(", "describe.skip(", "it(", "test(", "test.describe("] {
        if indent <= 2 && s.starts_with(kw) {
            let lit: String = s[kw.len()..].trim_start_matches(['\'', '"', '`']).chars().take_while(|c| !matches!(c, '\'' | '"' | '`')).take(70).collect();
            return Some(format!("{}{}", kw.trim_end_matches('('), if lit.is_empty() { String::new() } else { format!(" {lit}") }));
        }
    }
    None
}

fn sql_symbol(t: &str) -> Option<String> {
    let u = t.to_ascii_uppercase();
    let hit = ["CREATE ", "ALTER TABLE", "DROP ", "GRANT ", "REVOKE ", "COMMENT ON", "DO $$"].iter().any(|p| u.starts_with(p));
    hit.then(|| t.trim_end_matches(['(', ' ']).chars().take(80).collect())
}

fn prisma_symbol(t: &str) -> Option<String> {
    ["model ", "enum ", "datasource ", "generator ", "view "].iter().any(|k| t.starts_with(k)).then(|| t.trim_end_matches(['{', ' ']).to_string())
}

fn tf_symbol(t: &str) -> Option<String> {
    ["resource ", "data ", "variable ", "module ", "output ", "locals", "provider ", "terraform"].iter().any(|k| t.starts_with(k)).then(|| t.trim_end_matches(['{', ' ']).chars().take(80).collect())
}

fn shell_symbol(t: &str) -> Option<String> {
    if let Some(r) = t.strip_prefix("function ").or_else(|| t.strip_prefix("Function ")) {
        return Some(format!("function {}", ident_after(r).trim_end_matches('{')));
    }
    let n = t.split("()").next().unwrap_or("");
    (t.contains("()") && !n.is_empty() && n.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-')).then(|| format!("function {n}"))
}

fn symbol_of(lang: Lang, line: &str) -> Option<String> {
    let t = line.trim_start();
    if t.is_empty() {
        return None;
    }
    let indent = indent_of(line);
    match lang {
        Lang::Rust if indent <= 4 => rust_symbol(t),
        Lang::Py if indent <= 4 => py_symbol(t),
        Lang::Ts if indent <= 4 => ts_symbol(t, indent),
        Lang::Sql if indent == 0 => sql_symbol(t),
        Lang::Prisma if indent == 0 => prisma_symbol(t),
        Lang::Tf if indent == 0 => tf_symbol(t),
        Lang::Shell if indent <= 2 => shell_symbol(t),
        _ => None,
    }
}

fn heading_of(line: &str) -> Option<String> {
    let t = line.trim_end();
    let hashes = t.bytes().take_while(|b| *b == b'#').count();
    ((1..=6).contains(&hashes) && t[hashes..].starts_with(' ')).then(|| t[hashes..].trim().chars().take(100).collect())
}

fn is_lead_in(lang: Lang, line: &str) -> bool {
    let t = line.trim_start();
    match lang {
        Lang::Rust => t.starts_with("///") || t.starts_with("#[") || t.starts_with("//"),
        Lang::Py => t.starts_with('#') || t.starts_with('@'),
        Lang::Ts => t.starts_with("//") || t.starts_with("/**") || t.starts_with('*') || t.starts_with("*/") || t.starts_with('@'),
        Lang::Sql => t.starts_with("--"),
        Lang::Tf | Lang::Shell | Lang::Prisma => t.starts_with('#') || t.starts_with("//"),
        _ => false,
    }
}

/// Window size for formats without symbols (json, yaml, toml, css...).
const WINDOW: usize = 40;

fn boundaries(lang: Lang, lines: &[&str]) -> Vec<(usize, String)> {
    let mut out: Vec<(usize, String)> = Vec::new();
    let mut fence = false;
    for (i, l) in lines.iter().enumerate() {
        if lang == Lang::Md {
            if l.trim_start().starts_with("```") {
                fence = !fence;
            }
            if let (false, Some(h)) = (fence, heading_of(l)) {
                out.push((i, h));
            }
            continue;
        }
        let Some(sym) = symbol_of(lang, l) else { continue };
        let last = out.last().map(|x| x.0).unwrap_or(0);
        let mut s = i;
        while s > last && is_lead_in(lang, lines[s - 1]) {
            s -= 1;
        }
        if out.is_empty() || s > last {
            out.push((s, sym));
        }
    }
    out
}

fn push_cut(out: &mut Vec<ChunkDraft>, lines: &[&str], a: usize, b: usize, title: &str) {
    let mut s = a;
    while s < b {
        let mut e = (s + MAX_LINES).min(b);
        let mut bytes = 0;
        for (k, l) in lines[s..e].iter().enumerate() {
            bytes += l.len() + 1;
            if bytes > MAX_BYTES && k > 0 {
                e = s + k;
                break;
            }
        }
        let text = lines[s..e].join("\n");
        if !text.trim().is_empty() {
            let text = if text.len() > MAX_BYTES { text.chars().take(MAX_BYTES).collect() } else { text };
            out.push(ChunkDraft { l0: s as u32 + 1, l1: e as u32, title: title.to_string(), text });
        }
        s = e;
    }
}

/// Name declared on this line (`pub const X: ..`, `fn f(`, `class C`, `export const y =`, `NAME = ...` in Python): the definition site of an identifier.
pub fn def_name_of_line(line: &str) -> Option<String> {
    let mut t = line.trim_start();
    for w in ["pub(crate) ", "pub(super) ", "pub ", "export ", "default ", "declare ", "async ", "unsafe ", "static ", "extern \"C\" "] {
        if let Some(r) = t.strip_prefix(w) {
            t = r.trim_start();
        }
    }
    for kw in ["const ", "fn ", "struct ", "enum ", "trait ", "type ", "mod ", "class ", "interface ", "def ", "function ", "let ", "var ", "static ", "union ", "macro_rules! "] {
        if let Some(r) = t.strip_prefix(kw) {
            let r = r.trim_start().strip_prefix("mut ").unwrap_or(r.trim_start());
            let n = ident_after(r);
            return (n.len() >= 2).then(|| n.to_ascii_lowercase());
        }
    }
    let n = ident_after(t);
    let rest = t[n.len()..].trim_start();
    let upper = n.len() >= 3 && n.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
    (upper && (rest.starts_with('=') && !rest.starts_with("==") || rest.starts_with(':'))).then(|| n.to_ascii_lowercase())
}

pub fn chunk_file(path: &str, text: &str) -> Vec<ChunkDraft> {
    let lang = lang_of(path);
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }
    let b = boundaries(lang, &lines);
    let mut out = Vec::new();
    let mut starts: Vec<(usize, String)> = Vec::new();
    if b.first().map(|x| x.0 > 0).unwrap_or(true) {
        starts.push((0, "(top)".to_string()));
    }
    starts.extend(b);
    if lang == Lang::Other && starts.len() == 1 {
        for (k, _) in (0..lines.len()).step_by(WINDOW).enumerate() {
            let a = k * WINDOW;
            push_cut(&mut out, &lines, a, (a + WINDOW).min(lines.len()), "(window)");
        }
        return out;
    }
    for (k, (a, title)) in starts.iter().enumerate() {
        let e = starts.get(k + 1).map(|x| x.0).unwrap_or(lines.len());
        push_cut(&mut out, &lines, *a, e, title);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_symbols_with_doc_comments() {
        let src = "use std::io;\n\n/// Does a thing.\n/// More.\npub fn alpha(x: u32) -> u32 {\n    x + 1\n}\n\nimpl Foo for Bar {\n    pub fn beta(&self) {\n        let _ = 1;\n    }\n}\n";
        let c = chunk_file("src/x.rs", src);
        let titles: Vec<_> = c.iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, vec!["(top)", "fn alpha", "impl Foo for Bar", "fn beta"], "{c:?}");
        assert_eq!(c[1].l0, 3, "doc comment attaches to its symbol");
        assert!(c[1].text.contains("Does a thing"));
    }

    #[test]
    fn markdown_headings_ignore_fenced_code() {
        let md = "# Title\nintro\n```\n# not a heading\n```\n## Second\nbody\nmore\n";
        let c = chunk_file("docs/a.md", md);
        assert_eq!(c.iter().map(|c| c.title.as_str()).collect::<Vec<_>>(), vec!["Title", "Second"]);
        assert_eq!((c[0].l0, c[0].l1, c[1].l0), (1, 5, 6));
    }

    #[test]
    fn python_and_ts_symbols() {
        let py = "import os\n\n\nclass A:\n    def m(self):\n        return 1\n\n    def n(self):\n        return 2\n";
        assert!(chunk_file("a.py", py).iter().any(|c| c.title == "def n"));
        let ts = "import x from 'y';\n\nexport async function withTenant(id: string) {\n  return 1;\n}\n\nexport const RATE = 5;\n";
        let t: Vec<_> = chunk_file("a.ts", ts).into_iter().map(|c| c.title).collect();
        assert!(t.contains(&"function withTenant".to_string()), "{t:?}");
    }

    #[test]
    fn long_segments_are_cut_and_other_files_windowed() {
        let long: String = (0..200).map(|i| format!("line {i}\n")).collect();
        let c = chunk_file("data.yaml", &long);
        assert!(c.len() >= 5 && c.iter().all(|c| ((c.l1 - c.l0) as usize) < MAX_LINES));
        assert_eq!(c.last().unwrap().l1, 200);
    }
}
