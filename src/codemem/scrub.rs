//! Port of ingest/scrub.py (scrub-v1) for index chunks. Differences, on purpose: (1) matches are REDACTED in place and the chunk is kept
//! (the Python policy drops a whole record; for code, dropping a chunk would lose the pointer), (2) the sensitive-path-MENTION redaction is
//! not applied (a doc that names `.env.example` is not a secret; sensitive FILES are excluded from the index before reading, see `walk`),
//! (3) Rust `regex` has no look-around, so the two look-around patterns are re-expressed with capture groups.

use regex::{Captures, Regex};
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub const SCRUB_VERSION: &str = "codemem-scrub-1 (port of scrub-v1)";

struct Rx {
    secrets: Vec<(&'static str, Regex)>,
    password: Regex,
    placeholder: Regex,
    card: Regex,
    entropy: Regex,
    email: Regex,
    phone: Regex,
    uuid: Regex,
    hex: Regex,
}

fn rx() -> &'static Rx {
    static R: OnceLock<Rx> = OnceLock::new();
    R.get_or_init(|| {
        let c = |s: &str| Regex::new(s).expect("scrub regex");
        Rx {
            secrets: vec![
                ("private_key", c(r"-----BEGIN[ A-Z0-9]*PRIVATE KEY-----[\s\S]*?(?:-----END[ A-Z0-9]*PRIVATE KEY-----|\z)")),
                ("jwt", c(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}")),
                ("api_key", c(concat!(
                    r"\b(?:sk-ant-[A-Za-z0-9_-]{16,}|sk-(?:proj-|live-|test-)?[A-Za-z0-9_-]{20,}|(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{12,}",
                    r"|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|glpat-[A-Za-z0-9_-]{16,}",
                    r"|xox[abposr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{30,}|npm_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|pypi-[A-Za-z0-9_-]{30,}",
                    r"|SG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}|ya29\.[A-Za-z0-9_-]{20,}|dop_v1_[a-f0-9]{40,}|shpat_[a-fA-F0-9]{20,})"
                ))),
                ("bearer", c(r"(?i)\b(?:bearer|token)\s+[A-Za-z0-9_\-.=+/]{16,}")),
                ("conn_string", c(r#"(?i)\b[a-z][a-z0-9+.-]{1,20}://[^\s:/@"'`]+:[^\s@"'`]+@[^\s"'`]+"#)),
            ],
            password: c(concat!(
                r#"(?i)(?P<k>\b(?:password|passwd|pwd|pass|secret|api[_-]?key|apikey|access[_-]?key|secret[_-]?key|client[_-]?secret"#,
                r#"|auth[_-]?token|access[_-]?token|refresh[_-]?token|private[_-]?key|session[_-]?secret|signing[_-]?key)\b["']?\s*(?:=>|=|:)\s*)"#,
                r#"(?P<q>["'`]?)(?P<v>[^\s"'`,;)}\]]{6,})"#
            )),
            placeholder: c(concat!(
                r"(?i)^(?:<[^>]*>|\$\{?[A-Za-z_][\w.]*\}?|process\.env\.[\w.]+|os\.environ.*|env\(.*|getenv.*|x{3,}|\*{3,}|\.{3,}|changeme|placeholder|example|",
                r"redacted|\[redacted.*|none|null|undefined|true|false|string|str|number|optional|required|z\.\w+\(?.*|string\(\)|",
                r"\w+\(\)|[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+|self\.\w+|this\.\w+|input|value|args?\b.*)$"
            )),
            card: c(r"\b(?:\d[ -]?){13,19}\b"),
            entropy: c(r"[A-Za-z0-9+/=_-]{24,}"),
            email: c(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            phone: c(r"(^|[^\w+])(\+\d{1,3}[\s-]?(?:\d[\s-]?){7,12}\d\b)"),
            uuid: c(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
            hex: c(r"^[0-9a-fA-F]+$"),
        }
    })
}

fn shannon(s: &str) -> f64 {
    let mut counts: BTreeMap<char, usize> = BTreeMap::new();
    let mut n = 0usize;
    for ch in s.chars() {
        *counts.entry(ch).or_default() += 1;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    counts.values().map(|&c| { let p = c as f64 / n as f64; -p * p.log2() }).sum()
}

fn looks_high_entropy(tok: &str) -> bool {
    let r = rx();
    if tok.len() < 24 || r.uuid.is_match(tok) || r.hex.is_match(tok) {
        return false;
    }
    let t = tok.trim_matches('=');
    let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
    let upper = t.chars().filter(|c| c.is_uppercase()).count();
    let lower = t.chars().filter(|c| c.is_lowercase()).count();
    if digits < 3 || upper < 2 || lower < 2 {
        return false;
    }
    let seps = t.chars().filter(|c| matches!(c, '-' | '_' | '/')).count();
    if seps as f64 > t.chars().count() as f64 / 6.0 {
        return false;
    }
    shannon(t) >= 4.2
}

fn luhn_ok(digits: &str) -> bool {
    let (mut s, mut alt) = (0u32, false);
    for ch in digits.chars().rev() {
        let mut d = ch as u32 - 48;
        if alt {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        s += d;
        alt = !alt;
    }
    s % 10 == 0
}

/// Counts per category (never the matched content).
pub type Hits = BTreeMap<&'static str, usize>;

fn bump(h: &mut Hits, k: &'static str, n: usize) {
    if n > 0 {
        *h.entry(k).or_default() += n;
    }
}

fn redact_password(s: &str, hits: &mut Hits) -> String {
    let r = rx();
    let mut n = 0;
    let out = r.password.replace_all(s, |m: &Captures| {
        let v = &m["v"];
        let lower = v.to_ascii_lowercase();
        if r.placeholder.is_match(v) || ((lower.starts_with("ci-only-") || lower.starts_with("local-dev-")) && shannon(v) < 3.0) {
            return m[0].to_string();
        }
        n += 1;
        format!("{}{}[REDACTED:password]", &m["k"], &m["q"])
    });
    bump(hits, "password", n);
    out.into_owned()
}

fn redact_cards(s: &str, hits: &mut Hits) -> String {
    let mut n = 0;
    let out = rx().card.replace_all(s, |m: &Captures| {
        let d: String = m[0].chars().filter(|c| c.is_ascii_digit()).collect();
        let distinct = d.chars().collect::<std::collections::BTreeSet<_>>().len();
        if (13..=19).contains(&d.len()) && luhn_ok(&d) && distinct > 1 {
            n += 1;
            "[REDACTED:card]".to_string()
        } else {
            m[0].to_string()
        }
    });
    bump(hits, "card_number", n);
    out.into_owned()
}

/// Redacts secrets and PII in `text`; returns the scrubbed text and per-category counts.
pub fn scrub(text: &str) -> (String, Hits) {
    let r = rx();
    let mut hits = Hits::new();
    let mut s = text.to_string();
    for (cat, re) in &r.secrets {
        let n = re.find_iter(&s).count();
        if n > 0 {
            s = re.replace_all(&s, format!("[REDACTED:{cat}]").as_str()).into_owned();
            bump(&mut hits, cat, n);
        }
    }
    s = redact_password(&s, &mut hits);
    s = redact_cards(&s, &mut hits);
    let mut ne = 0;
    s = r.entropy.replace_all(&s, |m: &Captures| {
        if looks_high_entropy(&m[0]) { ne += 1; "[REDACTED:high_entropy]".to_string() } else { m[0].to_string() }
    }).into_owned();
    bump(&mut hits, "high_entropy", ne);
    let n = r.email.find_iter(&s).count();
    if n > 0 {
        s = r.email.replace_all(&s, "[REDACTED:email]").into_owned();
        bump(&mut hits, "email", n);
    }
    let n = r.phone.captures_iter(&s).count();
    if n > 0 {
        s = r.phone.replace_all(&s, "${1}[REDACTED:phone]").into_owned();
        bump(&mut hits, "phone", n);
    }
    (s, hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_keys_conn_strings_and_pii() {
        let (s, h) = scrub("k = sk-ant-api03-abcdefghijklmnopqrstuv\nurl postgres://app:hunter2pw@db.example.com/x\nmail me@corp.io tel +47 912 34 567");
        assert!(!s.contains("abcdefghijklmnop") && !s.contains("hunter2pw") && !s.contains("me@corp.io") && !s.contains("912 34"), "{s}");
        assert!(h.contains_key("api_key") && h.contains_key("conn_string") && h.contains_key("email") && h.contains_key("phone"));
    }

    #[test]
    fn password_assignments_redacted_but_placeholders_and_code_kept() {
        let (s, h) = scrub("password: \"s3cretValue99\"\npassword = process.env.DB_PASSWORD\npassword: z.string()\napiKey: ci-only-aaaaaa");
        assert!(s.contains("[REDACTED:password]") && !s.contains("s3cretValue99"), "{s}");
        assert!(s.contains("process.env.DB_PASSWORD") && s.contains("z.string()") && s.contains("ci-only-aaaaaa"), "{s}");
        assert_eq!(h.get("password"), Some(&1));
    }

    #[test]
    fn private_key_block_and_high_entropy() {
        let (s, h) = scrub("-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----\nx = Zk3Qv9Lm2Xa7Tw4Rp8Ys1Nb6Hd5Jc0Fe");
        assert!(!s.contains("MIIabc") && !s.contains("Zk3Qv9"), "{s}");
        assert!(h.contains_key("private_key") && h.contains_key("high_entropy"));
    }

    #[test]
    fn ordinary_code_untouched() {
        let code = "export async function withTenant(tenantId: string) { return set_config('app.tenant_id', tenantId, true); }";
        let (s, h) = scrub(code);
        assert_eq!(s, code);
        assert!(h.is_empty());
    }
}
