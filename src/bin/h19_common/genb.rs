//! Paper (P), business (B) and control generators.
use super::gens::*;
use super::*;

// ---------------------------------------------------------------- PAPER
pub struct Met {
    pub id: &'static str,
    pub dev_hdr: &'static str,
    pub un_hdr: &'static [&'static str],
    /// display multiplier in unseen header variant `un_scale` (display = canon / un_scale for variants flagged)
    pub unit: &'static str,
    pub lower: bool,
    pub lo: f64,
    pub hi: f64,
    pub dec: usize,
    pub dev_name: &'static str,
    pub un_name: &'static str,
}
pub const METS: &[Met] = &[
    Met { id: "acc", dev_hdr: "Accuracy (%)", un_hdr: &["Top-1 acc. (%)", "Accuracy (%)"], unit: "%", lower: false, lo: 60.0, hi: 96.0, dec: 1, dev_name: "accuracy", un_name: "top-1 accuracy" },
    Met { id: "f1", dev_hdr: "F1 (%)", un_hdr: &["F-score (%)", "F1 (%)"], unit: "%", lower: false, lo: 55.0, hi: 95.0, dec: 1, dev_name: "F1", un_name: "F-score" },
    Met { id: "bleu", dev_hdr: "BLEU", un_hdr: &["BLEU-4", "BLEU"], unit: "pts", lower: false, lo: 18.0, hi: 42.0, dec: 1, dev_name: "BLEU", un_name: "BLEU score" },
    Met { id: "lat", dev_hdr: "Latency (ms)", un_hdr: &["Latency (s)", "Inference time (s)", "Latency (ms)"], unit: "ms", lower: true, lo: 8.0, hi: 900.0, dec: 0, dev_name: "latency", un_name: "inference time" },
    Met { id: "params", dev_hdr: "Params (M)", un_hdr: &["Model size (B params)", "Params (M)"], unit: "M", lower: true, lo: 5.0, hi: 900.0, dec: 1, dev_name: "parameter count", un_name: "model size" },
    Met { id: "mem", dev_hdr: "Memory (GB)", un_hdr: &["Mem. footprint (GB)", "Memory (GB)"], unit: "GB", lower: true, lo: 1.0, hi: 40.0, dec: 1, dev_name: "memory", un_name: "memory footprint" },
    Met { id: "err", dev_hdr: "Error rate (%)", un_hdr: &["Err. (%)", "Error rate (%)"], unit: "%", lower: true, lo: 3.0, hi: 30.0, dec: 1, dev_name: "error rate", un_name: "misclassification rate" },
];
pub const MODELS: &[&str] = &["ResNet-50", "BERT-base", "OurNet", "ViT-B", "T5-small", "DistilNet", "LSTM-Base", "MobileNet", "GPT-mini", "Tiny-Former", "ConvMixer", "Perceiver"];
pub const PARAMS: &[(&str, &str, &str)] = &[
    // id, dev phrase, unseen phrase ({v} marks the value)
    ("bs", "a batch size of {v}", "mini-batches of {v} samples"),
    ("ep", "{v} epochs", "{v} passes over the training data"),
    ("lr", "a learning rate of {v}", "a step size of {v}"),
    ("n", "{v} participants", "a cohort of {v} subjects"),
    ("dropout", "a dropout rate of {v}", "dropout probability {v}"),
    ("gpus", "{v} GPUs", "{v} accelerators"),
];

fn fmt_dec(v: f64, dec: usize) -> String {
    format!("{:.*}", dec, v)
}

pub fn gen_paper(seed: u64, idx: u64, bank: Bank, layout: u8) -> Doc {
    let bk = if bank == Bank::Dev { 0 } else { 1 };
    let mut r = R::new(seed, 21, idx, bk);
    let unseen = bank == Bank::Unseen;
    let nm = r.range(4, 6) as usize;
    let mut pool: Vec<&str> = MODELS.to_vec();
    r.shuffle(&mut pool);
    let mut models: Vec<&str> = pool[..nm].to_vec();
    if !models.contains(&"OurNet") {
        models[0] = "OurNet";
    }
    let nmet = r.range(3, 4) as usize;
    let mut mi: Vec<usize> = (0..METS.len()).collect();
    r.shuffle(&mut mi);
    let mut mets: Vec<&Met> = mi[..nmet].iter().map(|&i| &METS[i]).collect();
    if !mets.iter().any(|m| !m.lower) {
        mets[0] = &METS[0];
    }
    // values (canonical)
    let mut canon: Vec<Vec<f64>> = Vec::new(); // [metric][model]
    for m in &mets {
        let mut vs: Vec<f64> = Vec::new();
        loop {
            vs.clear();
            for _ in 0..nm {
                let v = m.lo + (m.hi - m.lo) * r.f();
                vs.push(format!("{:.*}", m.dec, v).parse::<f64>().unwrap());
            }
            let mut s = vs.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if s.windows(2).all(|w| (w[1] - w[0]).abs() > 1e-9) {
                break;
            }
        }
        canon.push(vs);
    }
    let style = match layout {
        0 => TStyle::Pipe,
        1 => TStyle::Aligned(false),
        2 => TStyle::Csv,
        3 => TStyle::Pipe,
        _ => TStyle::Aligned(false),
    };
    let transposed = layout >= 4;
    let pm = layout >= 3;
    let mixed_units = layout >= 4;
    // header display decisions per metric
    struct Disp {
        hdr: String,
        div: f64,
        dec: usize,
    }
    let mut disp: Vec<Disp> = Vec::new();
    for m in &mets {
        let mut hdr = m.dev_hdr.to_string();
        let mut div = 1.0;
        let mut dec = m.dec;
        if unseen && r.chance(0.7) {
            hdr = (*r.pick(m.un_hdr)).to_string();
        }
        if hdr.contains("(s)") {
            div = 1000.0;
            dec = 4;
        }
        if hdr.contains("(B params)") {
            div = 1000.0;
            dec = 3;
        }
        if layout >= 3 && !unseen && r.chance(0.0) {
            hdr = hdr.split(" (").next().unwrap().to_string();
        }
        if mixed_units && m.id == "lat" {
            hdr = "Latency".into();
            div = 1.0;
            dec = 0;
        }
        disp.push(Disp { hdr, div, dec });
    }
    let mut b = B::new();
    let mut facts: Vec<Fact> = Vec::new();
    let mut claims: Vec<Claim> = Vec::new();
    let title_words = ["Efficient", "Scalable", "Robust", "Compact", "Adaptive"];
    b.line(&format!("{} {} for Benchmark Tasks", *r.pick(&title_words), if r.chance(0.5) { "Training" } else { "Inference" }));
    b.nl();
    b.line("Abstract");
    let best_acc = models.iter().position(|&m| m == "OurNet").unwrap();
    b.line(&format!("We present OurNet and evaluate it against {} baselines on {} datasets in {}. Code and data are available at release {}.", nm - 1, r.range(2, 9), r.range(2015, 2024), r.range(1, 4)));
    b.nl();
    b.line("1. Methods");
    // params
    let mut pids: Vec<usize> = (0..PARAMS.len()).collect();
    r.shuffle(&mut pids);
    let np = r.range(2, 4) as usize;
    let mut sent = String::from("We trained all models with ");
    let mut pieces: Vec<(usize, String)> = Vec::new();
    for &pi in &pids[..np] {
        let (id, dev, un) = PARAMS[pi];
        let phr = if unseen && r.chance(0.7) { un } else { dev };
        let val: f64 = match id {
            "bs" => (1u64 << r.range(4, 10)) as f64,
            "ep" => r.range(5, 300) as f64,
            "lr" => *r.pick(&[1e-4, 3e-4, 1e-3, 5e-5, 2e-5, 3e-3]),
            "n" => r.range(60, 4800) as f64,
            "dropout" => *r.pick(&[0.1, 0.2, 0.3, 0.5]),
            _ => r.range(1, 16) as f64,
        };
        let vs = match id {
            "lr" => {
                if r.chance(0.5) { format!("{:e}", val).replace("e-", "e-") } else { format!("{}", val) }
            }
            "dropout" => format!("{}", val),
            "n" => fmt_mag(val as i64, Sep::Us),
            _ => format!("{}", val as i64),
        };
        pieces.push((pi, phr.replace("{v}", &format!("\u{1}{}\u{2}", vs))));
        let _ = id;
    }
    let np_ = pieces.len();
    for (k, (_, p)) in pieces.iter().enumerate() {
        if k > 0 {
            sent.push_str(if k + 1 == np_ { ", and " } else { ", " });
        }
        sent.push_str(p);
    }
    sent.push_str(&format!(" (random seed {}; see Appendix {} and [{}]).", r.range(1, 99), (b'A' + r.u(4) as u8) as char, r.range(1, 40)));
    // emit with span extraction of \x01..\x02 markers
    let mut cur = 0usize;
    let mut pi_iter = pieces.iter().map(|x| x.0).collect::<Vec<_>>().into_iter();
    let sb = sent.as_bytes();
    let mut out = String::new();
    let base = b.pos();
    let mut marks: Vec<(usize, usize)> = Vec::new();
    while cur < sb.len() {
        if sb[cur] == 1 {
            let st = out.len();
            cur += 1;
            while sb[cur] != 2 {
                out.push(sb[cur] as char);
                cur += 1;
            }
            marks.push((base + st, base + out.len()));
            cur += 1;
        } else {
            // copy a full utf8 char
            let ch = sent[cur..].chars().next().unwrap();
            out.push(ch);
            cur += ch.len_utf8();
        }
    }
    b.line(&out);
    for (k, pi) in pi_iter.by_ref().enumerate() {
        let id = PARAMS[pi].0;
        let (lo, hi) = marks[k];
        let txt = &b.s[lo..hi];
        let v: f64 = txt.replace(',', "").parse().unwrap();
        facts.push(Fact::num(&format!("par|{}", id), FKind::Param, v, "", lo, hi));
    }
    b.nl();
    b.line("2. Results");
    b.line(&format!("Table {}: Main results on the held-out test set.{}", r.range(1, 4), if layout >= 3 { " Best per column in bold; values after the plus-minus sign are standard deviations over 5 runs." } else { "" }));
    b.nl();
    // table
    let best_idx: Vec<usize> = mets
        .iter()
        .enumerate()
        .map(|(j, m)| {
            let vs = &canon[j];
            let mut bi = 0;
            for i in 1..nm {
                if (m.lower && vs[i] < vs[bi]) || (!m.lower && vs[i] > vs[bi]) {
                    bi = i;
                }
            }
            bi
        })
        .collect();
    let cell_txt = |r: &mut R, j: usize, i: usize| -> String {
        let m = mets[j];
        let d = &disp[j];
        let v = canon[j][i] / d.div;
        let mut t = fmt_dec(v, d.dec);
        if mixed_units && m.id == "lat" {
            t = if canon[j][i] >= 1000.0 { format!("{:.2} s", canon[j][i] / 1000.0) } else { format!("{} ms", canon[j][i] as i64) };
        }
        if pm && !m.lower && m.id != "lat" {
            t = format!("{} \u{b1} {:.1}", t, 0.1 + 0.9 * r.f());
        }
        if pm && best_idx[j] == i {
            t = format!("**{}**", t);
        }
        t
    };
    let mut spans_flat: Vec<(usize, usize, usize, usize)> = Vec::new(); // (metric, model, lo, hi)
    if !transposed {
        let mut header = vec!["Model".to_string()];
        header.extend(disp.iter().map(|d| d.hdr.clone()));
        let mut rows = Vec::new();
        for i in 0..nm {
            let cells: Vec<String> = (0..nmet).map(|j| cell_txt(&mut r, j, i)).collect();
            rows.push(Row { label: models[i].to_string(), cells });
        }
        let sp = render_table(&mut b, style, &header, &rows);
        for i in 0..nm {
            for j in 0..nmet {
                spans_flat.push((j, i, sp[i][j].0, sp[i][j].1));
            }
        }
    } else {
        let mut header = vec!["Metric".to_string()];
        header.extend(models.iter().map(|m| m.to_string()));
        let mut rows = Vec::new();
        for j in 0..nmet {
            let cells: Vec<String> = (0..nm).map(|i| cell_txt(&mut r, j, i)).collect();
            rows.push(Row { label: disp[j].hdr.clone(), cells });
        }
        let sp = render_table(&mut b, style, &header, &rows);
        for j in 0..nmet {
            for i in 0..nm {
                spans_flat.push((j, i, sp[j][i].0, sp[j][i].1));
            }
        }
    }
    for (j, i, lo, hi) in &spans_flat {
        facts.push(Fact::num(&format!("cell|{}|{}", models[*i], mets[*j].id), FKind::Cell, canon[*j][*i], mets[*j].unit, *lo, *hi));
    }
    b.nl();
    // prose duplicates (consistent) + distractors
    {
        let j = 0;
        let m = mets[j];
        let i = best_idx[j];
        if !(mixed_units && m.id == "lat") && disp[j].div == 1.0 && r.chance(0.6) {
            b.line(&format!("{} reaches {} {} in this setting (p < 0.0{}, n = {}).", models[i], fmt_dec(canon[j][i], disp[j].dec), if m.unit == "%" { "%" } else { m.unit }, r.range(1, 5), r.range(20, 400)));
            b.nl();
        }
    }
    b.line("3. Discussion");
    // claims
    let ncl = r.range(3, 5) as usize;
    let mut n_sents = 0;
    let mut used: Vec<String> = Vec::new();
    let mut tries = 0;
    while claims.len() < ncl && tries < 40 {
        tries += 1;
        let j = r.u(nmet);
        let m = mets[j];
        let a = r.u(nm);
        let mut bb = r.u(nm);
        if bb == a {
            bb = (bb + 1) % nm;
        }
        let pos = r.chance(0.65);
        let hedged = r.chance(0.3);
        // support: want consistency with prob .7
        let better = |x: usize, y: usize| if m.lower { canon[j][x] < canon[j][y] } else { canon[j][x] > canon[j][y] };
        let want_support = r.chance(0.7);
        let (mut a, mut bb) = (a, bb);
        let supp = |a: usize, bb: usize| if pos { better(a, bb) } else { !better(a, bb) };
        if supp(a, bb) != want_support {
            std::mem::swap(&mut a, &mut bb);
        }
        let key = format!("clm|{}|{}|{}", models[a], models[bb], m.id);
        let key_rev = format!("clm|{}|{}|{}", models[bb], models[a], m.id);
        if used.contains(&key) || used.contains(&key_rev) {
            continue;
        }
        used.push(key.clone());
        let cite = r.range(1, 40) as u32;
        let (pv, nv): (&[&str], &[&str]) = if unseen && r.chance(0.6) { (&["beats", "surpasses", "yields gains over", "is superior to"], &["falls short of", "trails", "is no better than", "does not surpass"]) } else { (&["outperforms", "improves on", "is better than"], &["does not outperform", "fails to improve on", "is not better than"]) };
        let verb = if pos { *r.pick(pv) } else { *r.pick(nv) };
        let mname = if unseen && r.chance(0.6) { m.un_name } else { m.dev_name };
        let hedge_dev = ["Our results suggest that ", "It appears that ", "It may be that "];
        let hedge_un = ["We conjecture that ", "It is arguably true that ", "It seems likely that "];
        let hp: String = if hedged { if unseen && r.chance(0.6) { (*r.pick(&hedge_un)).to_string() } else { (*r.pick(&hedge_dev)).to_string() } } else { String::new() };
        let lower = |t: &str| -> String { let mut c = t.chars(); match c.next() { Some(f) => f.to_lowercase().collect::<String>() + c.as_str(), None => String::new() } };
        let structure = if unseen && r.chance(0.35) && pos { 2 } else { r.u(2) };
        let s = match structure {
            0 => format!("{}{} {} {} in terms of {} [{}].", hp, models[a], verb, models[bb], mname, cite),
            1 => format!("On {}, {}{} {} {} [{}].", mname, lower(&hp), models[a], verb, models[bb], cite),
            _ => format!("Compared with {}, {}{} {} on {} [{}].", models[bb], lower(&hp), models[a], verb, mname, cite),
        };
        let clo = b.pos();
        b.line(&s);
        n_sents += 1;
        let supported = supp(a, bb);
        claims.push(Claim { key: key.clone(), a: models[a].to_string(), b: models[bb].to_string(), metric: m.id.to_string(), pos, hedged, cite, supported });
        facts.push(Fact { key, kind: FKind::Claim, value: if pos { 1.0 } else { -1.0 }, sval: String::new(), unit: format!("{}|{}", if hedged { "hedged" } else { "firm" }, cite), lo: clo, hi: clo + s.len() });
    }
    b.line(&format!("Prior work [{}] used a single seed; our protocol follows [{}].", r.range(1, 40), r.range(1, 40)));
    b.nl();
    b.line("4. Limitations");
    b.line(&format!("We evaluate on {} datasets only and did not test beyond {} tokens; effects may not transfer to other regimes.", r.range(2, 9), fmt_mag(r.range(128, 8192), Sep::Us)));
    b.nl();
    b.line("References");
    for k in 1..=r.range(3, 6) {
        b.line(&format!("[{}] A. Author and B. Writer. Journal of Machine Studies {}({}), pp. {}-{}, {}.", k, r.range(3, 40), r.range(1, 12), r.range(10, 400), r.range(401, 900), r.range(2012, 2023)));
    }
    // questions
    let mut qs: Vec<Question> = Vec::new();
    for (j, m) in mets.iter().enumerate().take(2) {
        let keys: Vec<String> = (0..nm).map(|i| format!("cell|{}|{}", models[i], m.id)).collect();
        let vs: Vec<f64> = (0..nm).map(|i| canon[j][i]).collect();
        qs.push(Question { op: QOp::ArgBest(m.lower), keys, truth: qeval(QOp::ArgBest(m.lower), &vs).unwrap(), text: format!("Which model is best on {}?", m.dev_name) });
    }
    {
        let j = r.u(nmet);
        let (a, bb) = (0usize, 1usize);
        let keys = vec![format!("cell|{}|{}", models[a], mets[j].id), format!("cell|{}|{}", models[bb], mets[j].id)];
        let vs = vec![canon[j][a], canon[j][bb]];
        qs.push(Question { op: QOp::Diff, keys: keys.clone(), truth: qeval(QOp::Diff, &vs).unwrap(), text: format!("Difference in {} between {} and {}?", mets[j].dev_name, models[a], models[bb]) });
        qs.push(Question { op: QOp::Ratio, keys, truth: qeval(QOp::Ratio, &vs).unwrap(), text: "Ratio?".into() });
    }
    let firm: Vec<&Claim> = claims.iter().filter(|c| !c.hedged).collect();
    if firm.len() >= 2 {
        let keys: Vec<String> = firm.iter().map(|c| c.key.clone()).collect();
        let truth = firm.iter().filter(|c| c.supported).count() as f64;
        qs.push(Question { op: QOp::CountSupported, keys, truth, text: "How many firm claims are supported by the tables?".into() });
    }
    let _ = best_acc;
    let scale_conf = mixed_units || disp.iter().any(|d| d.div != 1.0) || pm;
    finalize(Doc { world: World::Paper, bank, layout, adv: 0, text: b.s, facts, claims, qs, n_distract: 0, scale_conf, structured: true, n_claim_sents: n_sents })
}

// ---------------------------------------------------------------- BUSINESS
const PARTIES: &[&str] = &["Acme Corp", "Beta Logistics LLC", "Cobalt Systems Inc.", "Delta Foods Ltd", "Evergreen Partners", "Fjord Analytics AB", "Granite Holdings", "Helios Energy plc"];

fn ordinal(d: i64) -> String {
    let s = match (d % 10, d % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{}{}", d, s)
}
/// format a date; returns (text, style id)
fn fmt_date(r: &mut R, bank: Bank, days: i64, with_weekday: bool) -> String {
    let (y, m, d) = civil_from_days(days);
    let mname = MONTHS[(m - 1) as usize];
    let style = if bank == Bank::Dev || r.chance(0.35) { r.u(3) } else { 3 + r.u(4) };
    let core = match style {
        0 => format!("{} {}, {}", mname, d, y),
        1 => format!("{:04}-{:02}-{:02}", y, m, d),
        2 => format!("{:02}/{:02}/{}", m, d, y),
        3 => format!("{} day of {}, {}", ordinal(d), mname, y),
        4 => format!("{} {} {}", d, mname, y),
        5 => format!("{:02}.{:02}.{}", d, m, y),
        _ => format!("{:02}/{:02}/{}", d, m, y),
    };
    if with_weekday && style == 0 {
        format!("{}, {}", WEEKDAYS[weekday(days)], core)
    } else {
        core
    }
}
fn fmt_amount(r: &mut R, bank: Bank, amt: i64, cur: &str) -> String {
    let unseen = bank == Bank::Unseen && r.chance(0.65);
    let sym = match cur {
        "USD" => "$",
        _ => "\u{20ac}",
    };
    if !unseen {
        match r.u(3) {
            0 => format!("{}{}", sym, fmt_mag(amt, Sep::Us)),
            1 => format!("{}{}.00", sym, fmt_mag(amt, Sep::Us)),
            _ => format!("{} {}", cur, fmt_mag(amt, Sep::Us)),
        }
    } else {
        match r.u(4) {
            0 if amt % 1000 == 0 => format!("{}k {}", amt / 1000, cur),
            1 => format!("{} {}", cur, fmt_mag(amt, Sep::Space)),
            2 => format!("\u{20ac}{},00", fmt_mag(amt, Sep::Eu)),
            _ => format!("{} {}", fmt_mag(amt, Sep::Us), if cur == "USD" { "US dollars" } else { "euros" }),
        }
    }
}
fn parse_amount_text_value(s: &str) -> f64 {
    // value of the digits in the span (only used for numeric truth bookkeeping; generator knows the true value anyway)
    s.chars().filter(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0.0)
}

pub fn gen_biz(seed: u64, idx: u64, bank: Bank, layout: u8) -> Doc {
    let mut r = R::new(seed, 31, idx, if bank == Bank::Dev { 0 } else { 1 });
    if r.chance(0.5) { gen_contract(&mut r, bank, layout) } else { gen_kpi(&mut r, bank, layout) }
}

fn gen_contract(r: &mut R, bank: Bank, layout: u8) -> Doc {
    let unseen = bank == Bank::Unseen;
    let mut b = B::new();
    let mut facts: Vec<Fact> = Vec::new();
    let p1 = *r.pick(PARTIES);
    let mut p2 = *r.pick(PARTIES);
    while p2 == p1 {
        p2 = *r.pick(PARTIES);
    }
    let cur = if unseen && r.chance(0.25) { "EUR" } else { "USD" };
    let eff = days_from_civil(r.range(2019, 2025), r.range(1, 12), r.range(1, 28));
    let fee_m: i64 = r.range(8, 400) * 250; // monthly
    let (per_txt, mult): (String, f64) = {
        let k = r.u(3);
        let dev = ["per month", "per quarter", "per annum"];
        let un_ = ["on a monthly basis", "every quarter", "each year"];
        let txt = if unseen && r.chance(0.6) { un_[k] } else { dev[k] };
        (txt.to_string(), [1.0, 3.0, 12.0][k])
    };
    let fee_amt = (fee_m as f64 * mult) as i64;
    let term_m: i64 = *r.pick(&[12, 24, 36, 6, 18]);
    let notice: i64 = *r.pick(&[30, 60, 90, 15, 45]);
    let int_pct = *r.pick(&[1.0, 1.5, 2.0, 0.5, 8.0, 12.0]);
    let cap: i64 = r.range(2, 80) * 5000;
    b.line("SERVICES AGREEMENT");
    b.nl();
    let wd = layout >= 2 && r.chance(0.4);
    let dtxt = fmt_date(r, bank, eff, wd);
    let kv_parties = unseen && r.chance(0.6);
    if kv_parties {
        b.push("This Services Agreement (the \"Agreement\") is made on ");
        let (lo, hi) = b.span(&dtxt);
        b.line(".");
        facts.push(Fact::num("date|effective", FKind::Date, eff as f64, "", lo, hi));
        b.push("Provider: ");
        let (lo, hi) = b.span(p1);
        b.line(".");
        facts.push(Fact { key: "party|supplier".into(), kind: FKind::Party, value: 0.0, sval: p1.into(), unit: "".into(), lo, hi });
        b.push("Client: ");
        let (lo, hi) = b.span(p2);
        b.line(".");
        facts.push(Fact { key: "party|customer".into(), kind: FKind::Party, value: 0.0, sval: p2.into(), unit: "".into(), lo, hi });
    } else {
        b.push("This Services Agreement (the \"Agreement\") is entered into as of ");
        let (lo, hi) = b.span(&dtxt);
        facts.push(Fact::num("date|effective", FKind::Date, eff as f64, "", lo, hi));
        b.push(" between ");
        let (lo, hi) = b.span(p1);
        facts.push(Fact { key: "party|supplier".into(), kind: FKind::Party, value: 0.0, sval: p1.into(), unit: "".into(), lo, hi });
        b.push(" (\"Supplier\") and ");
        let (lo, hi) = b.span(p2);
        facts.push(Fact { key: "party|customer".into(), kind: FKind::Party, value: 0.0, sval: p2.into(), unit: "".into(), lo, hi });
        b.line(" (\"Customer\").");
    }
    b.line(&format!("Notices shall be sent to 1{} Market Street, Suite {}, tel. (415) 555-0{}.", r.range(10, 99), r.range(100, 900), r.range(100, 199)));
    b.nl();
    // term
    let (term_n, term_unit) = if term_m % 12 == 0 && r.chance(0.5) { (term_m / 12, "years") } else { (term_m, "months") };
    let use_dev_term = !unseen || r.chance(0.4);
    b.push("1. Term. ");
    if use_dev_term {
        b.push("The initial term of this Agreement is ");
        if r.chance(0.5) {
            b.push(&format!("{} (", if term_n == 12 { "twelve" } else if term_n == 24 { "twenty-four" } else { "several" }));
            let (lo, hi) = b.span(&term_n.to_string());
            b.push(&format!(") {}", term_unit));
            facts.push(Fact::num("term_months", FKind::Clause, term_m as f64, "months", lo, hi));
        } else {
            let (lo, hi) = b.span(&term_n.to_string());
            b.push(&format!(" {}", term_unit));
            facts.push(Fact::num("term_months", FKind::Clause, term_m as f64, "months", lo, hi));
        }
        b.line(", commencing on the Effective Date.");
    } else {
        let s = match r.u(3) {
            0 => ("The Agreement shall run for ", format!(" calendar {}", term_unit), "."),
            1 => ("The duration of this Agreement is ", format!(" {}", term_unit), "."),
            _ => ("This Agreement continues for an initial period of ", format!(" {}", term_unit), "."),
        };
        b.push(s.0);
        let (lo, hi) = b.span(&term_n.to_string());
        b.push(&s.1);
        b.line(s.2);
        facts.push(Fact::num("term_months", FKind::Clause, term_m as f64, "months", lo, hi));
    }
    // fees
    let setup_first = r.chance(0.5);
    let setup = if r.chance(0.35) { Some(r.range(2, 20) * 250) } else { None };
    let setup_word = if unseen && r.chance(0.6) { "initial implementation" } else { "one-time setup" };
    if let (Some(s), true) = (setup, setup_first) {
        b.line(&format!("   A {} fee of {} is payable on signature.", setup_word, fmt_amount(r, Bank::Dev, s, cur)));
    }
    b.push("2. Fees. ");
    let fee_dev = !unseen || r.chance(0.4);
    let atxt = fmt_amount(r, bank, fee_amt, cur);
    if fee_dev {
        b.push("The Customer shall pay the Supplier a fee of ");
    } else {
        b.push("The Client shall remunerate the Provider ");
    }
    let (lo, hi) = b.span(&atxt);
    b.push(&format!(" {}", per_txt));
    if r.chance(0.45) {
        let tcv = fee_m * term_m;
        b.push(&format!(" (i.e., a total contract value of {} over the term)", fmt_amount(r, Bank::Dev, tcv, cur)));
    }
    b.line(".");
    facts.push(Fact::num("fee_monthly", FKind::Amount, fee_m as f64, cur, lo, hi));
    if let (Some(s), false) = (setup, setup_first) {
        b.line(&format!("   A {} fee of {} is payable on signature.", setup_word, fmt_amount(r, Bank::Dev, s, cur)));
    }
    // termination
    b.push("3. Termination. Either party may terminate this Agreement upon ");
    let dev_notice = !unseen || r.chance(0.4);
    let (lo, hi) = b.span(&notice.to_string());
    if dev_notice {
        b.line(" days' written notice.");
    } else {
        b.line(&format!(" calendar days of notice to the other party."));
    }
    facts.push(Fact::num("notice_days", FKind::Clause, notice as f64, "days", lo, hi));
    // interest
    b.push("4. Late Payment. Overdue amounts bear interest at ");
    if !unseen || r.chance(0.5) {
        let (lo, hi) = b.span(&format!("{}", int_pct));
        b.line("% per annum.");
        facts.push(Fact::num("int_pct", FKind::Clause, int_pct, "pct", lo, hi));
    } else {
        let (lo, hi) = b.span(&format!("{}", int_pct));
        b.line(" percent per year.");
        facts.push(Fact::num("int_pct", FKind::Clause, int_pct, "pct", lo, hi));
    }
    // liability
    b.push("5. Liability. The Supplier's aggregate liability ");
    b.push(if !unseen || r.chance(0.5) { "shall not exceed " } else { "is limited to " });
    let ctxt = fmt_amount(r, bank, cap, cur);
    let (lo, hi) = b.span(&ctxt);
    b.line(".");
    facts.push(Fact::num("cap", FKind::Amount, cap as f64, cur, lo, hi));
    b.line(&format!("6. Survival. Section {}.{} survives termination for {} months; uptime shall be at least 99.{}% (Exhibit {}).", r.range(2, 9), r.range(1, 6), r.range(6, 36), r.range(5, 9), (b'A' + r.u(3) as u8) as char));
    // facts spans for amounts need numeric values: truth known
    let _ = parse_amount_text_value;
    let mut qs = Vec::new();
    let g = |k: &str, facts: &Vec<Fact>| facts.iter().find(|f| f.key == k).map(|f| f.value).unwrap();
    let (fm, tm, ef) = (g("fee_monthly", &facts), g("term_months", &facts), g("date|effective", &facts));
    qs.push(Question { op: QOp::Prod, keys: vec!["fee_monthly".into(), "term_months".into()], truth: fm * tm, text: "Total contract value over the term?".into() });
    qs.push(Question { op: QOp::EndDate, keys: vec!["date|effective".into(), "term_months".into()], truth: qeval(QOp::EndDate, &[ef, tm]).unwrap(), text: "End date of the initial term?".into() });
    qs.push(Question { op: QOp::Lookup, keys: vec!["cap".into()], truth: g("cap", &facts), text: "Liability cap?".into() });
    qs.push(Question { op: QOp::Lookup, keys: vec!["notice_days".into()], truth: g("notice_days", &facts), text: "Notice period (days)?".into() });
    let scale_conf = wd || per_txt.contains("annum") || per_txt.contains("year") || per_txt.contains("quarter") || setup.is_some() || cur == "EUR";
    finalize(Doc { world: World::Biz, bank, layout, adv: 0, text: b.s, facts, claims: vec![], qs, n_distract: 0, scale_conf, structured: true, n_claim_sents: 0 })
}

struct Kpi {
    id: &'static str,
    dev: &'static str,
    un: &'static str,
    unit: &'static str, // USD, pct, pts, count
    lo: i64,
    hi: i64,
}
const KPIS: &[Kpi] = &[
    Kpi { id: "mrr", dev: "MRR", un: "Monthly recurring revenue", unit: "USD", lo: 200, hi: 9000 },
    Kpi { id: "churn", dev: "Churn rate", un: "Customer attrition", unit: "pct", lo: 5, hi: 90 },
    Kpi { id: "nps", dev: "NPS", un: "Net promoter score", unit: "pts", lo: 10, hi: 70 },
    Kpi { id: "cac", dev: "CAC", un: "Acquisition cost", unit: "USD", lo: 80, hi: 2500 },
    Kpi { id: "arpu", dev: "ARPU", un: "Avg revenue per user", unit: "USD", lo: 20, hi: 600 },
    Kpi { id: "headcount", dev: "Headcount", un: "Staff count", unit: "count", lo: 20, hi: 900 },
    Kpi { id: "ltv", dev: "LTV", un: "Lifetime value", unit: "USD", lo: 500, hi: 9000 },
];
pub fn kpi_dev_labels() -> Vec<(&'static str, &'static str)> {
    KPIS.iter().map(|k| (k.id, k.dev)).collect()
}

fn fmt_money(r: &mut R, unseen: bool, base: i64) -> String {
    // base in USD (multiple of 1000 when >=1000)
    if base >= 1_000_000 && base % 10_000 == 0 {
        let s = format!("{:.2}", base as f64 / 1e6);
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if unseen && r.chance(0.6) { format!("${} mn", s) } else { format!("${}M", s) }
    } else if base >= 1000 && base % 1000 == 0 {
        if unseen && r.chance(0.5) { format!("${}k", base / 1000) } else { format!("${}K", base / 1000) }
    } else {
        format!("${}", fmt_mag(base, Sep::Us))
    }
}

fn gen_kpi(r: &mut R, bank: Bank, layout: u8) -> Doc {
    let unseen = bank == Bank::Unseen;
    let mut b = B::new();
    let mut facts: Vec<Fact> = Vec::new();
    let companies = ["Northwind SaaS", "Bluepeak Cloud", "Helix Platforms", "Orbit Apps"];
    let comp = *r.pick(&companies);
    let year = r.range(2022, 2025);
    let q = r.range(2, 4);
    let (pq, py) = (q - 1, year);
    let pd = |qq: i64, yy: i64, un: bool, r: &mut R| -> String {
        if un && r.chance(0.5) { if r.chance(0.5) { format!("{}Q{}", qq, yy % 100) } else { format!("Q{}'{}", qq, yy % 100) } } else { format!("Q{} {}", qq, yy) }
    };
    let per_key = |qq: i64, yy: i64| format!("Q{}-{}", qq, yy);
    let ptxt0 = pd(q, year, unseen, r);
    let ptxt1 = pd(pq, py, unseen, r);
    b.line(&format!("{} - {} KPI Report", comp, ptxt0));
    b.line("Prepared by Finance and Operations for the leadership team.");
    b.nl();
    let mut ids: Vec<usize> = (0..KPIS.len()).collect();
    r.shuffle(&mut ids);
    let nk = r.range(4, 6) as usize;
    let mut chosen: Vec<usize> = ids[..nk].to_vec();
    // ensure ltv+cac pairs are both present or none
    if chosen.contains(&6) && !chosen.contains(&3) {
        chosen.push(3);
    }
    if !chosen.contains(&0) {
        chosen.push(0);
    }
    if !chosen.contains(&1) {
        chosen.push(1);
    }
    chosen.sort();
    chosen.dedup();
    // values (current, previous) in base units / natural units
    let mut vals: Vec<(f64, f64)> = Vec::new();
    for &ki in &chosen {
        let k = &KPIS[ki];
        let a = r.range(k.lo, k.hi);
        let g = 0.85 + 0.35 * r.f();
        let p = ((a as f64 / g).round() as i64).max(k.lo);
        let conv = |x: i64| -> f64 {
            match k.unit {
                "USD" if k.id == "mrr" => (x * 1000) as f64,
                "pct" => x as f64 / 10.0,
                _ => x as f64,
            }
        };
        let (a, p) = if k.id == "mrr" { ((a / 50) * 50, (p / 50) * 50) } else { (a, p) };
        vals.push((conv(a), conv(p)));
    }
    let scale_hdr = layout >= 4;

    let fmt_val = |r: &mut R, k: &Kpi, v: f64| -> String {
        match k.unit {
            "USD" => {
                if scale_hdr && k.id == "mrr" { fmt_mag((v / 1000.0).round() as i64, Sep::Us) } else { fmt_money(r, unseen, v as i64) }
            }
            "pct" => format!("{:.1}%", v),
            _ => format!("{}", v as i64),
        }
    };
    let mut scale_conf = unseen || scale_hdr;
    if layout <= 1 || layout == 4 {
        // table with Change column
        let style = if layout == 1 { TStyle::Aligned(false) } else { TStyle::Pipe };
        let header = vec!["KPI".to_string(), ptxt0.clone(), ptxt1.clone(), "Change".to_string()];
        let mut rows = Vec::new();
        let mut meta = Vec::new();
        for (ci, &ki) in chosen.iter().enumerate() {
            let k = &KPIS[ki];
            let mut lab = (if unseen && r.chance(0.55) { k.un } else { k.dev }).to_string();
            if scale_hdr && k.id == "mrr" {
                lab = format!("{} (USD thousands)", lab);
            }
            let (a, p) = vals[ci];
            let ch = if k.unit == "pct" { format!("{:+.1} pp", a - p) } else { format!("{:+.1}%", (a - p) / p * 100.0) };
            let (sa, sp_) = (fmt_val(r, k, a), fmt_val(r, k, p));
            rows.push(Row { label: lab, cells: vec![sa, sp_, ch] });
            meta.push((k, a, p));
        }
        let sp = render_table(&mut b, style, &header, &rows);
        for (ri, (k, a, p)) in meta.iter().enumerate() {
            let unit = k.unit;
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(q, year)), FKind::Kpi, *a, unit, sp[ri][0].0, sp[ri][0].1));
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(pq, py)), FKind::Kpi, *p, unit, sp[ri][1].0, sp[ri][1].1));
        }
    } else if layout == 2 {
        for (ci, &ki) in chosen.iter().enumerate() {
            let k = &KPIS[ki];
            let lab = if unseen && r.chance(0.55) { k.un } else { k.dev };
            let (a, p) = vals[ci];
            b.push(&format!("- {}: ", lab));
            let (lo, hi) = b.span(&fmt_val(r, k, a));
            b.push(&format!(" ({}) vs ", ptxt0));
            let (lo2, hi2) = b.span(&fmt_val(r, k, p));
            b.line(&format!(" ({})", ptxt1));
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(q, year)), FKind::Kpi, a, k.unit, lo, hi));
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(pq, py)), FKind::Kpi, p, k.unit, lo2, hi2));
        }
    } else {
        // prose sentences
        for (ci, &ki) in chosen.iter().enumerate() {
            let k = &KPIS[ki];
            let lab = if unseen && r.chance(0.55) { k.un } else { k.dev };
            let (a, p) = vals[ci];
            let dev_phr = !unseen || r.chance(0.4);
            b.push(&format!("{} {} ", lab, if dev_phr { "was" } else { "came in at" }));
            let (lo, hi) = b.span(&fmt_val(r, k, a));
            if dev_phr {
                b.push(&format!(" in {}, {} from ", ptxt0, if a >= p { "up" } else { "down" }));
            } else {
                b.push(&format!(" for {} versus ", ptxt0));
            }
            let (lo2, hi2) = b.span(&fmt_val(r, k, p));
            if dev_phr {
                b.line(&format!(" in {}.", ptxt1));
            } else {
                b.line(" a quarter earlier.");
            }
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(q, year)), FKind::Kpi, a, k.unit, lo, hi));
            facts.push(Fact::num(&format!("kpi|{}|{}", k.id, per_key(pq, py)), FKind::Kpi, p, k.unit, lo2, hi2));
        }
    }
    b.nl();
    // commentary with consistent growth statement + distractors
    let mrr_i = chosen.iter().position(|&k| k == 0).unwrap();
    let (a, p) = vals[mrr_i];
    let nd = days_from_civil(year, 11, r.range(1, 28));
    b.line(&format!("Commentary: MRR grew {:.1}% quarter over quarter; we closed {} enterprise deals and hired {} people. Next review is on {}.", (a / p - 1.0) * 100.0, r.range(2, 40), r.range(1, 25), fmt_date(r, Bank::Dev, nd, false)));
    scale_conf |= layout >= 3;
    // questions
    let mut qs = Vec::new();
    let kk = |id: &str, qq: i64, yy: i64| format!("kpi|{}|{}", id, per_key(qq, yy));
    let tv = |key: &str, facts: &Vec<Fact>| facts.iter().find(|f| f.key == key).map(|f| f.value);
    if let (Some(x), Some(y)) = (tv(&kk("mrr", q, year), &facts), tv(&kk("mrr", pq, py), &facts)) {
        qs.push(Question { op: QOp::Growth, keys: vec![kk("mrr", q, year), kk("mrr", pq, py)], truth: (x - y) / y, text: "MRR growth quarter over quarter?".into() });
    }
    if let (Some(x), Some(y)) = (tv(&kk("churn", q, year), &facts), tv(&kk("churn", pq, py), &facts)) {
        qs.push(Question { op: QOp::Diff, keys: vec![kk("churn", q, year), kk("churn", pq, py)], truth: x - y, text: "Change in churn (pp)?".into() });
    }
    if let (Some(x), Some(y)) = (tv(&kk("ltv", q, year), &facts), tv(&kk("cac", q, year), &facts)) {
        qs.push(Question { op: QOp::Ratio, keys: vec![kk("ltv", q, year), kk("cac", q, year)], truth: x / y, text: "LTV/CAC?".into() });
    }
    if let Some(x) = tv(&kk("mrr", q, year), &facts) {
        qs.push(Question { op: QOp::Lookup, keys: vec![kk("mrr", q, year)], truth: x, text: "Current MRR?".into() });
    }
    finalize(Doc { world: World::Biz, bank, layout, adv: 0, text: b.s, facts, claims: vec![], qs, n_distract: 0, scale_conf, structured: true, n_claim_sents: 0 })
}

// ---------------------------------------------------------------- controls
pub fn gen_nostruct(seed: u64, idx: u64, bank: Bank) -> Doc {
    let mut r = R::new(seed, 41, idx, 0);
    let _ = bank;
    let topics = ["the onboarding process", "the vendor selection", "the migration plan", "the support rota", "the hiring plan"];
    let t = *r.pick(&topics);
    let mut b = B::new();
    b.line(&format!("Meeting notes - {}", t));
    for _ in 0..r.range(5, 9) {
        let who = *r.pick(&["Dana", "Marco", "Priya", "Lee", "Sven"]);
        let what = *r.pick(&["raised a concern about", "agreed to own", "asked to postpone", "proposed a pilot for", "will follow up on"]);
        b.line(&format!("{} {} {} before the {} review (about {} people were present).", who, what, t, ordinal(r.range(2, 29)), r.range(4, 30)));
    }
    b.line(&format!("Decision: the team will proceed with {} and revisit it at the next sync.", t));
    let q = Question { op: QOp::Qual, keys: vec![], truth: 1.0, text: "What was decided?".into() };
    finalize(Doc { world: World::NoStruct, bank: Bank::Dev, layout: 0, adv: 0, text: b.s, facts: vec![], claims: vec![], qs: vec![q], n_distract: 0, scale_conf: false, structured: false, n_claim_sents: 0 })
}
pub fn gen_tiny(seed: u64, idx: u64) -> Doc {
    let mut r = R::new(seed, 43, idx, 0);
    let amt = r.range(5, 400) * 25;
    let mut b = B::new();
    b.push("Invoice: total due ");
    let (lo, hi) = b.span(&format!("${}", fmt_mag(amt, Sep::Us)));
    b.line(" on receipt.");
    let facts = vec![Fact::num("fee_monthly", FKind::Amount, amt as f64, "USD", lo, hi)];
    let q = Question { op: QOp::Lookup, keys: vec!["fee_monthly".into()], truth: amt as f64, text: "Amount due?".into() };
    finalize(Doc { world: World::Tiny, bank: Bank::Dev, layout: 0, adv: 0, text: b.s, facts, claims: vec![], qs: vec![q], n_distract: 0, scale_conf: false, structured: true, n_claim_sents: 0 })
}

/// generate a document of `world` for (seed, idx); layout from the world's mix; bank by `unseen_p`
pub fn gen_doc(world: World, seed: u64, idx: u64, unseen_p: f64, layout_w: &[f64; 5]) -> Doc {
    let mut r = R::new(seed, 99, idx, world.idx() as u64);
    let bank = if r.chance(unseen_p) { Bank::Unseen } else { Bank::Dev };
    let mut x = r.f();
    let mut layout = 4u8;
    for (i, w) in layout_w.iter().enumerate() {
        if x < *w {
            layout = i as u8;
            break;
        }
        x -= w;
    }
    match world {
        World::Paper => gen_paper(seed, idx, bank, layout),
        World::Fin => gen_fin(seed, idx, bank, layout),
        World::Biz => gen_biz(seed, idx, bank, layout),
        World::NoStruct => gen_nostruct(seed, idx, bank),
        World::Tiny => gen_tiny(seed, idx),
    }
}
