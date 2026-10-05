//! h18 rule-based spec EXTRACTOR: word problem text -> `fin` spec line + confidence, with abstention.
//! Lexicon = cue words auto-derived from the SEEN phrase bank (ask sentences + slot phrases); the UNSEEN bank uses other
//! words and frames, so unseen accuracy is the generalisation estimate. Method: tokenizer (money / percent / thousands /
//! number words / seeded-distractor rules) -> fuzzy word repair (Damerau <= 1, len >= 5) -> op candidates by idf-weighted
//! cue overlap -> per-candidate slot filling (lists, equations, units, enumerations, scalar numbers by exact bitmask DP
//! over cue-window scores) -> validity check by running the real tool -> multiplicative confidence.
use super::nl::*;
use neural_mesh::fin;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct Ex {
    pub spec: Option<String>,
    pub conf: f64,
    pub op: Option<&'static str>,
    pub fuzzy: usize,
    pub unused: usize,
    pub margin: f64,
}
impl Ex {
    fn none(fuzzy: usize) -> Ex {
        Ex { spec: None, conf: 0.0, op: None, fuzzy, unused: 0, margin: 0.0 }
    }
}

const STOP: &[&str] = &["the", "a", "an", "is", "are", "of", "at", "to", "in", "and", "it", "its", "be", "was", "were", "this", "that", "what", "how", "does", "do", "can", "will", "or", "by", "as", "on", "with", "for", "from", "you"];
fn is_stop(w: &str) -> bool {
    STOP.contains(&w)
}

// ---------------------------------------------------------------------------------------------
// lexicon

struct SlotLex {
    key: &'static str,
    kind: K,
    opt: bool,
    before: HashMap<String, f64>,
    after: HashMap<String, f64>,
}
struct OpLex {
    def: &'static OpDef,
    cues: HashSet<String>,
    slots: Vec<SlotLex>,
}
struct UnitLex {
    spec: &'static str,
    /// word sequence (lowercase) or a single case-sensitive symbol
    words: Vec<String>,
    symbol: bool,
}
struct Lex {
    ops: Vec<OpLex>,
    idf: HashMap<String, f64>,
    vocab: HashSet<String>,
    by_first: HashMap<(char, usize), Vec<String>>,
    units: Vec<UnitLex>,
}

fn clean_word(t: &str) -> String {
    let mut s = t.trim_start_matches(['(', '"', '\'']).trim_end_matches([')', '?', ',', ';', ':', '!', '"', '\'']).to_string();
    // trailing '.' unless it ends a decimal like "8." (never generated)
    while s.ends_with('.') {
        s.pop();
    }
    s
}
fn phrase_tokens(s: &str) -> Vec<String> {
    s.replace("{v}", " # ")
        .split_whitespace()
        .map(|t| {
            let c = clean_word(t);
            if c.starts_with('{') {
                if c.starts_with("{v}") || c.starts_with("#") {
                    "#".to_string()
                } else {
                    format!("@{}", c.trim_start_matches('{').split('}').next().unwrap_or(""))
                }
            } else {
                c.to_lowercase()
            }
        })
        .filter(|t| !t.is_empty())
        .collect()
}

fn lex() -> &'static Lex {
    static L: OnceLock<Lex> = OnceLock::new();
    L.get_or_init(|| {
        let mut vocab: HashSet<String> = HashSet::new();
        let mut ops: Vec<OpLex> = Vec::new();
        // boilerplate (lead-in / filler / distractor sentences of the SEEN bank) never counts as an op cue
        let mut noise: HashSet<String> = HashSet::new();
        for pool in LEADS_SEEN.iter() {
            for sx in pool.iter() {
                noise.extend(phrase_tokens(sx));
            }
        }
        for sx in FILLER_SEEN.iter().chain(DIST_SEEN.iter()) {
            noise.extend(phrase_tokens(sx));
        }
        for u in UNITS {
            for f in u.seen {
                for w in f.split_whitespace() {
                    vocab.insert(w.to_lowercase());
                }
            }
        }
        for def in OPS {
            let mut cues: HashSet<String> = HashSet::new();
            let noise_ref = &noise;
            let mut add_cues = |toks: &[String], cues: &mut HashSet<String>| {
                for (i, t) in toks.iter().enumerate() {
                    if t == "#" || t.starts_with('@') || is_stop(t) || noise_ref.contains(t) {
                        continue;
                    }
                    cues.insert(t.clone());
                    if i + 1 < toks.len() && toks[i + 1] != "#" && !toks[i + 1].starts_with('@') && !is_stop(&toks[i + 1]) && !noise_ref.contains(&toks[i + 1]) {
                        cues.insert(format!("{} {}", t, toks[i + 1]));
                    }
                }
            };
            for a in def.ask_seen {
                let t = phrase_tokens(a);
                add_cues(&t, &mut cues);
                for w in &t {
                    vocab.insert(w.clone());
                }
            }
            let mut slots = Vec::new();
            // slot-word counts for cue weights
            let mut word_slots: HashMap<String, usize> = HashMap::new();
            let mut per_slot: Vec<(HashSet<String>, HashSet<String>)> = def.slots.iter().map(|_| (HashSet::new(), HashSet::new())).collect();
            for (si, s) in def.slots.iter().enumerate() {
                for ph in s.seen {
                    let t = phrase_tokens(ph);
                    add_cues(&t, &mut cues);
                    // markers: "#" = this slot's own value, "@key" = another slot's value
                    let ismark = |w: &String| w == "#" || w.starts_with('@');
                    for (i, w) in t.iter().enumerate() {
                        if ismark(w) {
                            // owner of the marker
                            let owner = if w == "#" { Some(si) } else { def.slots.iter().position(|x| x.key == &w[1..]) };
                            if let Some(o) = owner {
                                let mut k = i;
                                while k > 0 && !ismark(&t[k - 1]) {
                                    k -= 1;
                                    if !is_stop(&t[k]) {
                                        per_slot[o].0.insert(t[k].clone());
                                        vocab.insert(t[k].clone());
                                    }
                                }
                                let mut k = i + 1;
                                while k < t.len() && !ismark(&t[k]) {
                                    if !is_stop(&t[k]) {
                                        per_slot[o].1.insert(t[k].clone());
                                        vocab.insert(t[k].clone());
                                    }
                                    k += 1;
                                }
                            }
                        }
                    }
                }
                for c in s.cue {
                    per_slot[si].0.insert(c.to_string());
                    per_slot[si].1.insert(c.to_string());
                    vocab.insert(c.to_string());
                }
                for e in s.en {
                    for sf in e.seen {
                        for w in phrase_tokens(sf) {
                            vocab.insert(w);
                        }
                    }
                }
            }
            for (bf, af) in &per_slot {
                let union: HashSet<&String> = bf.iter().chain(af.iter()).collect();
                for w in union {
                    *word_slots.entry(w.clone()).or_insert(0) += 1;
                }
            }
            for (i, s) in def.slots.iter().enumerate() {
                let (bf, af) = &per_slot[i];
                let w = |x: &String| 1.0 / (*word_slots.get(x).unwrap_or(&1) as f64);
                slots.push(SlotLex {
                    key: s.key,
                    kind: s.kind,
                    opt: s.opt,
                    before: bf.iter().map(|x| (x.clone(), w(x))).collect(),
                    after: af.iter().map(|x| (x.clone(), w(x))).collect(),
                });
            }
            ops.push(OpLex { def, cues, slots });
        }
        let n = ops.len() as f64;
        let mut df: HashMap<String, usize> = HashMap::new();
        for o in &ops {
            for c in &o.cues {
                *df.entry(c.clone()).or_insert(0) += 1;
            }
        }
        let idf = df.into_iter().map(|(k, v)| (k, (n / v as f64).ln() + 0.2)).collect();
        let mut by_first: HashMap<(char, usize), Vec<String>> = HashMap::new();
        for w in &vocab {
            if w.len() >= 4 && w.chars().all(|c| c.is_ascii_alphabetic()) {
                by_first.entry((w.chars().next().unwrap_or(' '), w.len())).or_default().push(w.clone());
            }
        }
        let mut units = Vec::new();
        for u in UNITS {
            for f in u.seen {
                let symbol = f.len() <= 4 && !f.contains(' ') && !f.chars().all(|c| c.is_ascii_lowercase() && f.len() > 2) || f.contains('/') || f.contains('^') || f.contains('*');
                let words: Vec<String> = if symbol { vec![f.to_string()] } else { f.split_whitespace().map(|w| w.to_lowercase()).collect() };
                units.push(UnitLex { spec: u.spec, words, symbol });
            }
        }
        // longest first
        units.sort_by(|a, b| b.words.len().cmp(&a.words.len()).then(b.words[0].len().cmp(&a.words[0].len())));
        Lex { ops, idf, vocab, by_first, units }
    })
}

fn damerau(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + c);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[n][m]
}
fn repair(w: &str, fuzzy: &mut usize) -> String {
    let l = lex();
    if w.len() < 5 || l.vocab.contains(w) || !w.chars().all(|c| c.is_ascii_lowercase()) {
        return w.to_string();
    }
    let first = w.chars().next().unwrap_or(' ');
    let mut best: Option<(&String, usize)> = None;
    let mut ties = 0;
    for len in [w.len() - 1, w.len(), w.len() + 1] {
        if let Some(c) = l.by_first.get(&(first, len)) {
            for v in c {
                let d = damerau(w, v);
                if d <= 1 {
                    match best {
                        Some((_, bd)) if bd < d => {}
                        Some((_, bd)) if bd == d => ties += 1,
                        _ => {
                            best = Some((v, d));
                            ties = 0;
                        }
                    }
                }
            }
        }
    }
    match best {
        Some((v, _)) if ties == 0 => {
            *fuzzy += 1;
            v.clone()
        }
        _ => w.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// tokens

#[derive(Clone, Debug)]
struct Tok {
    raw: String,
    low: String,
    num: Option<f64>,
    numraw: String,
    pct: bool,
    money: bool,
    wordnum: bool,
    sent: usize,
    ign: bool,
    used: bool,
}
const MONTHS: &[&str] = &["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"];

fn is_plain_number(c: &str) -> bool {
    let b = c.as_bytes();
    if b.is_empty() {
        return false;
    }
    let mut i = 0;
    if b[0] == b'-' {
        i = 1;
    }
    let (mut digits, mut dots) = (0, 0);
    while i < b.len() {
        if b[i].is_ascii_digit() {
            digits += 1;
        } else if b[i] == b'.' {
            dots += 1;
        } else {
            return false;
        }
        i += 1;
    }
    digits > 0 && dots <= 1
}
fn is_thousands(c: &str) -> bool {
    let t = c.trim_start_matches('-');
    let ip = t.split('.').next().unwrap_or("");
    let parts: Vec<&str> = ip.split(',').collect();
    parts.len() > 1 && parts[0].len() >= 1 && parts[0].len() <= 3 && parts[0].bytes().all(|b| b.is_ascii_digit()) && parts[1..].iter().all(|p| p.len() == 3 && p.bytes().all(|b| b.is_ascii_digit()))
}

fn tokenize(text: &str, fuzzy: &mut usize) -> Vec<Tok> {
    let mut toks: Vec<Tok> = Vec::new();
    let mut sent = 0usize;
    for rawtok in text.split_whitespace() {
        let ends_sent = (rawtok.ends_with('.') || rawtok.ends_with('?') || rawtok.ends_with('!')) && rawtok.len() > 1;
        let c = clean_word(rawtok);
        let mut t = Tok { raw: c.clone(), low: c.to_lowercase(), num: None, numraw: String::new(), pct: false, money: false, wordnum: false, sent, ign: false, used: false };
        if !c.is_empty() {
            let mut body = c.as_str();
            if let Some(b) = body.strip_prefix('$') {
                body = b;
                t.money = true;
            }
            if let Some(b) = body.strip_suffix('%') {
                body = b;
                t.pct = true;
            }
            let owned: String;
            if is_thousands(body) {
                owned = body.replace(',', "");
                body = &owned;
            }
            // 'minus' sign variants
            if is_plain_number(body) {
                if let Ok(v) = body.parse::<f64>() {
                    t.num = Some(v);
                    t.numraw = body.to_string();
                }
            } else if let Some(i) = NUMWORDS.iter().position(|w| *w == t.low) {
                t.num = Some(i as f64);
                t.numraw = i.to_string();
                t.wordnum = true;
            } else {
                t.low = repair(&t.low, fuzzy);
            }
        }
        if !t.low.is_empty() {
            toks.push(t);
        }
        if ends_sent {
            sent += 1;
        }
    }
    // post-pass: percent / money words, seeded distractor rules (SEEN distractor patterns only)
    let n = toks.len();
    for i in 0..n {
        if toks[i].num.is_none() {
            continue;
        }
        let next = toks.get(i + 1).map(|t| t.low.clone()).unwrap_or_default();
        let next2 = toks.get(i + 2).map(|t| t.low.clone()).unwrap_or_default();
        if next == "percent" || (next == "per" && next2 == "cent") {
            toks[i].pct = true;
        }
        if next == "dollars" {
            toks[i].money = true;
        }
        let prev = if i > 0 && toks[i - 1].sent == toks[i].sent { toks[i - 1].low.clone() } else { String::new() };
        if toks[i].wordnum {
            let counted = matches!(next.as_str(), "year" | "years" | "month" | "months" | "payments" | "monthly" | "instalments" | "installments" | "periods" | "units" | "significant") || matches!(prev.as_str(), "number" | "instalment" | "installment" | "payment");
            if !counted {
                toks[i].ign = true;
            }
        }
        if prev == "page" || prev == "ticket" || prev == "within" {
            toks[i].ign = true;
        }
        if MONTHS.contains(&prev.as_str()) {
            toks[i].ign = true;
            if i + 1 < n && toks[i + 1].raw.len() == 4 && toks[i + 1].num.is_some() {
                toks[i + 1].ign = true;
            }
        }
        if toks[i].pct && next == "confidence" {
            toks[i].ign = true;
        }
    }
    toks
}

// ---------------------------------------------------------------------------------------------
// slot filling

struct Fill {
    spec: String,
    quality: f64,
    unused: usize,
    weak: usize,
    ambig: usize,
}

fn sentence_has_numbers(toks: &[Tok], sent: usize) -> bool {
    toks.iter().any(|t| t.sent == sent && t.num.is_some() && !t.ign)
}

fn find_surface(toks: &[Tok], surface: &str) -> Option<usize> {
    let ws: Vec<String> = surface.split_whitespace().map(|w| w.to_lowercase()).collect();
    if ws.is_empty() || toks.len() < ws.len() {
        return None;
    }
    'o: for i in 0..=toks.len() - ws.len() {
        for (k, w) in ws.iter().enumerate() {
            if toks[i + k].low != *w || toks[i + k].sent != toks[i].sent {
                continue 'o;
            }
        }
        return Some(i);
    }
    None
}

fn pick_enum(toks: &[Tok], s: &Slot) -> Option<&'static str> {
    // surfaces must sit in a sentence without numbers
    let mut best: Option<(&'static str, usize)> = None;
    for e in s.en {
        for sf in e.seen {
            if let Some(i) = find_surface(toks, sf) {
                if sentence_has_numbers(toks, toks[i].sent) {
                    continue;
                }
                let l = sf.len();
                if best.map(|b| l > b.1).unwrap_or(true) {
                    best = Some((e.val, l));
                }
            }
        }
    }
    best.map(|b| b.0)
}

fn runs(toks: &[Tok]) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let mut last_num: Option<usize> = None;
    for (i, t) in toks.iter().enumerate() {
        if t.num.is_some() && !t.ign && !t.used && !t.pct {
            match last_num {
                Some(l) if toks[l].sent == t.sent && (i == l + 1 || (i == l + 2 && toks[l + 1].low == "and")) => cur.push(i),
                _ => {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    cur.push(i);
                }
            }
            last_num = Some(i);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn pick_cf_list(toks: &[Tok]) -> Option<(Vec<usize>, Vec<f64>)> {
    let rs = runs(toks);
    if rs.is_empty() {
        return None;
    }
    // outlay form: a single money number after an outlay cue, followed (same sentence) by a longer run
    for (ri, r) in rs.iter().enumerate() {
        if r.len() == 1 {
            let i = r[0];
            let cue = (i.saturating_sub(5)..i).any(|k| toks[k].sent == toks[i].sent && matches!(toks[k].low.as_str(), "investment" | "outlay" | "costs" | "cost" | "pays" | "spend"));
            if cue {
                if let Some(next) = rs.get(ri + 1) {
                    if toks[next[0]].sent == toks[i].sent {
                        let mut idx = vec![i];
                        idx.extend(next.iter());
                        let mut vals: Vec<f64> = vec![-toks[i].num?.abs()];
                        vals.extend(next.iter().map(|k| toks[*k].num.unwrap_or(0.0)));
                        return Some((idx, vals));
                    }
                }
            }
        }
    }
    let r = rs.iter().max_by_key(|r| r.len())?;
    if r.len() < 2 {
        return None;
    }
    Some((r.clone(), r.iter().map(|k| toks[*k].num.unwrap_or(0.0)).collect()))
}

fn parse_eq(toks: &mut [Tok]) -> Option<String> {
    let arrow = toks.iter().position(|t| t.low == "->" || t.low == "=>" || t.raw == "\u{2192}")?;
    let isspec = |t: &Tok| t.raw.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false) && fin::molar_mass_t(0, &t.raw).is_ok();
    let mut left: Vec<usize> = Vec::new();
    let mut i = arrow;
    while i > 0 {
        let k = i - 1;
        let t = &toks[k];
        if t.low == "+" || isspec(t) || (t.num.is_some() && !t.pct && t.numraw.chars().all(|c| c.is_ascii_digit())) {
            left.push(k);
            i -= 1;
        } else {
            break;
        }
    }
    left.reverse();
    let mut right: Vec<usize> = Vec::new();
    let mut j = arrow + 1;
    while j < toks.len() {
        let t = &toks[j];
        if t.low == "+" || isspec(t) || (t.num.is_some() && !t.pct && t.numraw.chars().all(|c| c.is_ascii_digit())) {
            right.push(j);
            j += 1;
            if toks[j - 1].sent != toks[arrow].sent {
                break;
            }
        } else {
            break;
        }
    }
    let side = |idx: &[usize], toks: &[Tok]| -> Option<String> {
        let mut out = String::new();
        let mut coef = String::new();
        for k in idx {
            let t = &toks[*k];
            if t.low == "+" {
                if !out.is_empty() && !out.ends_with('+') {
                    out.push('+');
                }
            } else if t.num.is_some() {
                coef = t.numraw.clone();
            } else {
                if coef != "1" {
                    out.push_str(&coef);
                }
                coef.clear();
                out.push_str(&t.raw);
            }
        }
        if out.is_empty() || out.ends_with('+') || !coef.is_empty() {
            None
        } else {
            Some(out)
        }
    };
    let (l, r) = (side(&left, toks)?, side(&right, toks)?);
    for k in left.iter().chain(right.iter()) {
        toks[*k].used = true;
    }
    toks[arrow].used = true;
    Some(format!("{l}>{r}"))
}

fn unit_mentions(toks: &[Tok]) -> Vec<(usize, usize, &'static str)> {
    let l = lex();
    let mut out: Vec<(usize, usize, &'static str)> = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let mut hit: Option<(usize, &'static str)> = None;
        for u in &l.units {
            let n = u.words.len();
            if i + n > toks.len() {
                continue;
            }
            let ok = (0..n).all(|k| {
                let t = &toks[i + k];
                t.sent == toks[i].sent && t.num.is_none() && if u.symbol { t.raw == u.words[0] } else { t.low == u.words[k] }
            });
            if ok {
                hit = Some((n, u.spec));
                break;
            }
        }
        if let Some((n, spec)) = hit {
            // the preposition "in" directly before a number is not the inch unit
            if spec == "in" && toks.get(i + 1).map(|t| t.num.is_some()).unwrap_or(false) {
                i += 1;
                continue;
            }
            out.push((i, n, spec));
            i += n;
        } else {
            i += 1;
        }
    }
    out
}

fn slot_scores(toks: &[Tok], cand: &[usize], sl: &SlotLex, rank_of_slot: f64) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    for (ci, &i) in cand.iter().enumerate() {
        let t = &toks[i];
        let mut ev = 0.0;
        let st = t.sent;
        // before window
        let mut dist = 0;
        let mut k = i;
        while k > 0 && dist < 6 {
            k -= 1;
            if toks[k].sent != st {
                break;
            }
            dist += 1;
            if let Some(w) = sl.before.get(&toks[k].low) {
                ev += *w / (1.0 + 0.3 * (dist as f64 - 1.0));
            }
        }
        // after window
        let mut dist = 0;
        let mut k = i + 1;
        while k < toks.len() && dist < 3 && toks[k].sent == st {
            dist += 1;
            let w = sl.after.get(&toks[k].low).or_else(|| sl.after.get(&toks[k].raw));
            if let Some(w) = w {
                ev += 0.8 * *w / (1.0 + 0.3 * (dist as f64 - 1.0));
            }
            k += 1;
        }
        // sentence level
        let mut far = 0.0;
        for kk in 0..toks.len() {
            if toks[kk].sent == st && kk != i && (sl.before.contains_key(&toks[kk].low) || sl.after.contains_key(&toks[kk].low)) {
                far += 0.05;
            }
        }
        ev += far;
        let mut kind = 0.0;
        match sl.kind {
            K::Rate | K::Pct => kind += if t.pct { 2.5 } else { -3.0 },
            K::Money => {
                if t.pct {
                    kind -= 3.0;
                }
                if t.money {
                    kind += 0.3;
                }
            }
            K::Count => {
                if t.pct {
                    kind -= 3.0;
                }
                if t.numraw.chars().all(|c| c.is_ascii_digit()) {
                    kind += 0.5;
                } else {
                    kind -= 1.5;
                }
            }
            _ => {
                if t.pct {
                    kind -= 3.0;
                }
            }
        }
        let rank_i = ci as f64 / (cand.len().max(2) - 1) as f64;
        let order = 0.15 * (1.0 - (rank_i - rank_of_slot).abs());
        out.push((ev * 2.0 + kind + order, ev));
    }
    out
}

/// exact assignment of distinct tokens to slots (bitmask DP). Required slots must be assigned; optional ones may be
/// skipped; every unassigned token costs `unused_pen`.
fn assign(scores: &[Vec<(f64, f64)>], optional: &[bool], ntok: usize, unused_pen: f64) -> Option<(Vec<Option<usize>>, f64)> {
    let s = scores.len();
    if ntok > 14 {
        return None;
    }
    let full = 1usize << ntok;
    let neg = f64::NEG_INFINITY;
    let mut dp = vec![vec![neg; full]; s + 1];
    let mut par: Vec<Vec<(usize, Option<usize>)>> = vec![vec![(0, None); full]; s + 1];
    dp[0][0] = 0.0;
    for j in 0..s {
        for mask in 0..full {
            let cur = dp[j][mask];
            if cur == neg {
                continue;
            }
            if optional[j] && cur > dp[j + 1][mask] {
                dp[j + 1][mask] = cur;
                par[j + 1][mask] = (mask, None);
            }
            for t in 0..ntok {
                if mask & (1 << t) == 0 {
                    let nm = mask | (1 << t);
                    let v = cur + scores[j][t].0;
                    if v > dp[j + 1][nm] {
                        dp[j + 1][nm] = v;
                        par[j + 1][nm] = (mask, Some(t));
                    }
                }
            }
        }
    }
    let mut best = neg;
    let mut bm = 0;
    for mask in 0..full {
        if dp[s][mask] == neg {
            continue;
        }
        let unused = ntok - (mask as u32).count_ones() as usize;
        let v = dp[s][mask] - unused_pen * unused as f64;
        if v > best {
            best = v;
            bm = mask;
        }
    }
    if best == neg {
        return None;
    }
    let mut res = vec![None; s];
    let mut mask = bm;
    for j in (1..=s).rev() {
        let (pm, t) = par[j][mask];
        res[j - 1] = t;
        mask = pm;
    }
    Some((res, best))
}

fn fmt_num(x: f64) -> String {
    fin::fmt(x)
}

fn fill(opx: &OpLex, base: &[Tok]) -> Option<Fill> {
    let def = opx.def;
    let mut toks: Vec<Tok> = base.to_vec();
    let mut vals: Vec<(&'static str, String)> = Vec::new();
    let mut weak = 0usize;
    let mut ambig = 0usize;
    // ---- enumerations first (they decide optional slots)
    let mut find_val: Option<&'static str> = None;
    for (i, s) in def.slots.iter().enumerate() {
        if s.kind == K::E {
            let v = pick_enum(&toks, s)?;
            if s.key == "find" {
                find_val = Some(v);
            }
            vals.push((opxkey(opx, i), v.to_string()));
        }
    }
    // ---- equation + species (stoich)
    let mut eq_txt: Option<String> = None;
    if def.slots.iter().any(|s| s.kind == K::Eq) {
        eq_txt = Some(parse_eq(&mut toks)?);
    }
    // ---- errprop pairs
    if def.op == "errprop" {
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        for i in 0..toks.len() {
            if toks[i].low == "+/-" || toks[i].raw == "\u{b1}" {
                let before = (0..i).rev().find(|k| toks[*k].num.is_some() && !toks[*k].ign)?;
                let after = (i + 1..toks.len()).find(|k| toks[*k].num.is_some() && !toks[*k].ign)?;
                pairs.push((before, after));
            }
        }
        if pairs.len() != 2 {
            return None;
        }
        for (k, (a, b)) in pairs.iter().enumerate() {
            let (kx, ks) = if k == 0 { ("x", "sx") } else { ("y", "sy") };
            vals.push((kx, fmt_num(toks[*a].num?)));
            vals.push((ks, fmt_num(toks[*b].num?)));
            toks[*a].used = true;
            toks[*b].used = true;
        }
    }
    // ---- units (uconv)
    if def.slots.iter().any(|s| s.kind == K::Unit) {
        let num_i = toks.iter().position(|t| t.num.is_some() && !t.ign && !t.used)?;
        let ms = unit_mentions(&toks);
        let from = ms.iter().find(|m| m.0 == num_i + 1)?;
        let to = ms.iter().find(|m| m.0 != from.0 && m.0 != num_i + 1)?;
        vals.push(("x", toks[num_i].numraw.clone()));
        vals.push(("from", from.2.to_string()));
        vals.push(("to", to.2.to_string()));
        toks[num_i].used = true;
    }
    // ---- formula
    if def.slots.iter().any(|s| s.kind == K::Formula) {
        let t = toks.iter().find(|t| {
            t.raw.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false)
                && (t.raw.chars().any(|c| c.is_ascii_digit()) || t.raw.chars().filter(|c| c.is_ascii_uppercase()).count() >= 2)
                && fin::molar_mass_t(0, &t.raw).is_ok()
        })?;
        vals.push(("f", t.raw.clone()));
    }
    // ---- species (stoich): given follows "<num> g of", want = another species of the equation in that sentence
    if let Some(eq) = &eq_txt {
        let (l, r) = eq.split_once('>')?;
        let sp: Vec<String> = l.split('+').chain(r.split('+')).map(|t| t.trim_start_matches(|c: char| c.is_ascii_digit()).to_string()).collect();
        let mut given: Option<String> = None;
        let mut gsent = 0usize;
        for i in 0..toks.len() {
            if toks[i].num.is_some() && !toks[i].used && i + 3 < toks.len() && matches!(toks[i + 1].low.as_str(), "g" | "grams") && toks[i + 2].low == "of" && sp.contains(&toks[i + 3].raw) {
                given = Some(toks[i + 3].raw.clone());
                gsent = toks[i + 3].sent;
                vals.push(("m", toks[i].numraw.clone()));
                toks[i].used = true;
                toks[i + 3].used = true;
                break;
            }
        }
        let given = given?;
        let sent = gsent;
        let want = toks.iter().find(|t| !t.used && t.sent == sent && sp.contains(&t.raw) && t.raw != given).map(|t| t.raw.clone())?;
        vals.push(("eq", eq.clone()));
        vals.push(("given", given));
        vals.push(("want", want));
    }
    // ---- lists
    let list_slots: Vec<usize> = def.slots.iter().enumerate().filter(|(_, s)| s.kind == K::List).map(|(i, _)| i).collect();
    if list_slots.len() == 1 {
        let (idx, v) = pick_cf_list(&toks)?;
        for k in idx {
            toks[k].used = true;
        }
        vals.push((def.slots[list_slots[0]].key, v.iter().map(|x| fmt_num(*x)).collect::<Vec<_>>().join(",")));
        if v.len() < 2 && def.op != "mean" {
            return None;
        }
    } else if list_slots.len() == 2 {
        let rs = runs(&toks);
        let mut rs: Vec<Vec<usize>> = rs.into_iter().filter(|r| r.len() >= 3).collect();
        if rs.len() < 2 {
            return None;
        }
        rs.truncate(2);
        // which is x: cue word 'x' / 'inputs' within 3 tokens before the run
        let is_x = |r: &Vec<usize>| -> i32 {
            let i = r[0];
            let mut sc = 0;
            for k in i.saturating_sub(4)..i {
                if toks[k].sent == toks[i].sent {
                    if toks[k].low == "x" {
                        sc += 2;
                    }
                    if toks[k].low == "y" {
                        sc -= 2;
                    }
                }
            }
            sc
        };
        let (a, b) = (is_x(&rs[0]), is_x(&rs[1]));
        let (xr, yr) = if b > a { (rs[1].clone(), rs[0].clone()) } else { (rs[0].clone(), rs[1].clone()) };
        if a == b {
            ambig += 1;
        }
        for (key, r) in [("x", &xr), ("y", &yr)] {
            vals.push((key, r.iter().map(|k| fmt_num(toks[*k].num.unwrap_or(0.0))).collect::<Vec<_>>().join(",")));
        }
        for k in xr.iter().chain(yr.iter()) {
            toks[*k].used = true;
        }
    }
    // ---- scalar numeric slots
    let scalar: Vec<usize> = def
        .slots
        .iter()
        .enumerate()
        .filter(|(_, s)| matches!(s.kind, K::Rate | K::Pct | K::Money | K::Count | K::Num) && !vals.iter().any(|(k, _)| *k == s.key) && !(def.op == "errprop"))
        .filter(|(_, s)| !(def.op == "stoich" && s.key == "m"))
        .filter(|(_, s)| !(def.op == "uconv" && s.key == "x"))
        .map(|(i, _)| i)
        .collect();
    let cand: Vec<usize> = toks.iter().enumerate().filter(|(_, t)| t.num.is_some() && !t.ign && !t.used).map(|(i, _)| i).collect();
    let mut unused_tokens = 0usize;
    if !scalar.is_empty() {
        // optional slots that are the find target are not present
        let active: Vec<usize> = scalar.into_iter().filter(|&i| Some(def.slots[i].key) != find_val || !def.slots[i].opt).collect();
        let active: Vec<usize> = active.into_iter().filter(|&i| !(def.slots[i].opt && find_val == Some(def.slots[i].key))).collect();
        if cand.len() < active.iter().filter(|&&i| !def.slots[i].opt).count() {
            return None;
        }
        let ns = active.len().max(2);
        let sc: Vec<Vec<(f64, f64)>> = active.iter().enumerate().map(|(r, &i)| slot_scores(&toks, &cand, &opx.slots[i], r as f64 / (ns - 1) as f64)).collect();
        let opt: Vec<bool> = active.iter().map(|&i| def.slots[i].opt).collect();
        let (asg, _best) = assign(&sc, &opt, cand.len(), 0.9)?;
        let mut assigned = 0;
        for (r, &i) in active.iter().enumerate() {
            let s = &def.slots[i];
            if let Some(ti) = asg[r] {
                assigned += 1;
                let t = &toks[cand[ti]];
                let (score, ev) = sc[r][ti];
                if ev < 0.2 {
                    weak += 1;
                }
                // a runner-up token within 0.5 of the best for this slot
                let second = sc[r].iter().enumerate().filter(|(k, _)| *k != ti).map(|(_, x)| x.0).fold(f64::NEG_INFINITY, f64::max);
                if second > score - 0.5 {
                    ambig += 1;
                }
                let v = t.num?;
                let text = match s.kind {
                    K::Rate => fmt_num(v / 100.0),
                    K::Pct | K::Money | K::Count => fmt_num(v),
                    _ => {
                        if def.op == "sfround" || def.op == "sci" {
                            t.numraw.clone()
                        } else {
                            fmt_num(v)
                        }
                    }
                };
                vals.push((s.key, text));
            } else if !s.opt {
                return None;
            }
        }
        unused_tokens = cand.len() - assigned;
    }
    // sfround: x then n, handled by scalar assignment? (x is Num, n is Count): both scalar -> done above
    // ---- assemble in the op's slot order
    let mut spec = def.op.to_string();
    for s in def.slots {
        if let Some((_, v)) = vals.iter().find(|(k, _)| *k == s.key) {
            spec.push(' ');
            spec.push_str(s.key);
            spec.push('=');
            spec.push_str(v);
        }
    }
    // the tool must accept it
    let a = fin::Args::parse(&spec).ok()?;
    if fin::eval(0, &a).is_err() {
        return None;
    }
    let tot_slots = def.slots.iter().filter(|s| !matches!(s.kind, K::Unit | K::Species)).count().max(1);
    let quality = (1.0 - 0.15 * weak as f64 / tot_slots as f64 * 3.0 - 0.1 * unused_tokens as f64).clamp(0.05, 1.0);
    Some(Fill { spec, quality, unused: unused_tokens, weak, ambig })
}
fn opxkey(opx: &OpLex, i: usize) -> &'static str {
    opx.slots[i].key
}

// ---------------------------------------------------------------------------------------------
// entry point

pub fn extract(text: &str) -> Ex {
    let l = lex();
    let mut fuzzy = 0usize;
    let toks = tokenize(text, &mut fuzzy);
    // text features
    let mut feats: HashSet<String> = HashSet::new();
    for (i, t) in toks.iter().enumerate() {
        if t.num.is_some() || is_stop(&t.low) {
            continue;
        }
        feats.insert(t.low.clone());
        if i + 1 < toks.len() && toks[i + 1].num.is_none() && !is_stop(&toks[i + 1].low) && toks[i + 1].sent == t.sent {
            feats.insert(format!("{} {}", t.low, toks[i + 1].low));
        }
    }
    let mut scored: Vec<(f64, usize)> = l
        .ops
        .iter()
        .enumerate()
        .map(|(oi, o)| {
            let s: f64 = o.cues.iter().filter(|c| feats.contains(*c)).map(|c| l.idf.get(c).copied().unwrap_or(1.0)).sum();
            (s / (o.cues.len() as f64).sqrt(), oi)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let smax = scored[0].0.max(1e-9);
    let mut best: Option<(f64, usize, Fill)> = None;
    let mut second_combined = 0.0f64;
    for (rank, (s, oi)) in scored.iter().take(6).enumerate() {
        if *s < 0.2 * smax {
            continue;
        }
        if let Some(f) = fill(&l.ops[*oi], &toks) {
            let combined = s / smax + 0.7 * f.quality - 0.03 * rank as f64;
            match &best {
                Some((bc, _, _)) if *bc >= combined => {
                    second_combined = second_combined.max(combined);
                }
                _ => {
                    if let Some((bc, _, _)) = &best {
                        second_combined = second_combined.max(*bc);
                    }
                    best = Some((combined, *oi, f));
                }
            }
        }
    }
    let (bc, oi, f) = match best {
        Some(b) => b,
        None => return Ex::none(fuzzy),
    };
    let margin = ((bc - second_combined) / bc.max(1e-9)).clamp(0.0, 1.0);
    let top_text = scored.iter().find(|(_, o)| *o == oi).map(|x| x.0 / smax).unwrap_or(0.0);
    let mut conf = 0.5 + 0.5 * margin.min(0.4) / 0.4;
    conf *= (0.55 + 0.45 * top_text).min(1.0);
    conf *= 0.8f64.powi(f.unused as i32);
    conf *= 0.93f64.powi(fuzzy.min(6) as i32);
    conf *= 0.82f64.powi(f.weak as i32);
    conf *= 0.85f64.powi(f.ambig.min(3) as i32);
    Ex { spec: Some(f.spec), conf: conf.clamp(0.0, 1.0), op: Some(l.ops[oi].def.op), fuzzy, unused: f.unused, margin }
}

/// confidence bin 0..4 (for selector features)
pub fn conf_bin(c: f64) -> usize {
    if c < 0.3 {
        0
    } else if c < 0.5 {
        1
    } else if c < 0.7 {
        2
    } else if c < 0.9 {
        3
    } else {
        4
    }
}

/// debug: text scores of the top ops and the fill result of each
pub fn debug(text: &str) -> String {
    let l = lex();
    let mut fuzzy = 0usize;
    let toks = tokenize(text, &mut fuzzy);
    let mut feats: HashSet<String> = HashSet::new();
    for (i, t) in toks.iter().enumerate() {
        if t.num.is_some() || is_stop(&t.low) {
            continue;
        }
        feats.insert(t.low.clone());
        if i + 1 < toks.len() && toks[i + 1].num.is_none() && !is_stop(&toks[i + 1].low) && toks[i + 1].sent == t.sent {
            feats.insert(format!("{} {}", t.low, toks[i + 1].low));
        }
    }
    let mut scored: Vec<(f64, usize)> = l
        .ops
        .iter()
        .enumerate()
        .map(|(oi, o)| {
            let s: f64 = o.cues.iter().filter(|c| feats.contains(*c)).map(|c| l.idf.get(c).copied().unwrap_or(1.0)).sum();
            (s / (o.cues.len() as f64).sqrt(), oi)
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut out = format!("tokens: {:?}\n", toks.iter().map(|t| format!("{}{}", t.low, if t.num.is_some() { format!("[{}{}{}s{}]", t.numraw, if t.pct { "%" } else { "" }, if t.ign { "!" } else { "" }, t.sent) } else { String::new() })).collect::<Vec<_>>());
    for (s, oi) in scored.iter().take(6) {
        let f = fill(&l.ops[*oi], &toks);
        out.push_str(&format!("{:<10} {:.3} fill={:?}\n", l.ops[*oi].def.op, s, f.map(|f| (f.spec, f.quality))));
    }
    out
}
