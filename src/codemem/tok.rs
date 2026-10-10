//! Identifier-aware tokenizer shared by indexing and querying. Whole identifiers are the base tokens (`[A-Za-z0-9_]+`, lowercased, as in
//! bench/frontend.py v2). `split` additionally emits snake_case / camelCase parts; RD-LOG R18/R55 rejected stemming and camelCase splitting on
//! doc text, so the default is OFF and the choice for code is measured separately (bench/codemem/README.md).

pub fn is_stop(t: &str) -> bool {
    matches!(
        t,
        "a" | "about" | "all" | "also" | "an" | "and" | "any" | "are" | "as" | "at" | "be" | "been" | "but" | "by" | "can" | "did" | "do"
            | "does" | "for" | "from" | "had" | "has" | "have" | "how" | "if" | "in" | "into" | "is" | "it" | "its" | "just" | "may" | "me"
            | "more" | "my" | "no" | "not" | "of" | "on" | "or" | "our" | "out" | "so" | "some" | "than" | "that" | "the" | "their"
            | "them" | "then" | "there" | "these" | "they" | "this" | "to" | "up" | "us" | "was" | "we" | "were" | "what" | "when"
            | "where" | "which" | "who" | "why" | "will" | "with" | "would" | "you" | "your" | "fn" | "pub" | "let" | "mut" | "def"
    )
}

fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn skip_token(t: &str) -> bool {
    let n = t.len();
    n < 2 || n > 48 || (t.bytes().all(|c| c.is_ascii_digit()) && n > 8) || (n >= 32 && t.bytes().all(|c| c.is_ascii_hexdigit()))
}

/// camelCase / snake_case parts of one identifier run (original case), lowercased, length >= 3.
fn parts(run: &str) -> Vec<String> {
    let mut out = Vec::new();
    for seg in run.split('_').filter(|s| !s.is_empty()) {
        let b = seg.as_bytes();
        let mut start = 0;
        for i in 1..b.len() {
            let lower_to_upper = b[i - 1].is_ascii_lowercase() && b[i].is_ascii_uppercase();
            let acronym_end = i + 1 < b.len() && b[i - 1].is_ascii_uppercase() && b[i].is_ascii_uppercase() && b[i + 1].is_ascii_lowercase();
            if lower_to_upper || acronym_end {
                out.push(seg[start..i].to_ascii_lowercase());
                start = i;
            }
        }
        out.push(seg[start..].to_ascii_lowercase());
    }
    out.retain(|p| p.len() >= 3 && !is_stop(p));
    out
}

pub fn tokenize(text: &str, split: bool, out: &mut Vec<String>) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !is_word(b[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < b.len() && is_word(b[i]) {
            i += 1;
        }
        let run = &text[s..i];
        let low = run.to_ascii_lowercase();
        if skip_token(&low) {
            continue;
        }
        if !is_stop(&low) {
            out.push(low.clone());
        }
        if split {
            let ps = parts(run);
            if ps.len() > 1 {
                out.extend(ps.into_iter().filter(|p| *p != low));
            }
        }
    }
}

pub fn tokens(text: &str, split: bool) -> Vec<String> {
    let mut v = Vec::new();
    tokenize(text, split, &mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_identifiers_and_stopwords() {
        assert_eq!(tokens("The withTenant_ctx() is N_DOMAINS 42", false), vec!["withtenant_ctx", "n_domains", "42"]);
    }

    #[test]
    fn split_adds_parts_only_when_enabled() {
        let t = tokens("withTenantContext HTTPServer snake_case_name", true);
        for want in ["withtenantcontext", "tenant", "context", "http", "server", "snake", "case", "name"] {
            assert!(t.iter().any(|x| x == want), "{want} missing in {t:?}");
        }
        assert!(!tokens("withTenantContext", false).contains(&"tenant".to_string()));
    }

    #[test]
    fn drops_hashes_and_long_numbers() {
        assert!(tokens("deadbeefdeadbeefdeadbeefdeadbeef 12345678901", false).is_empty());
    }
}
