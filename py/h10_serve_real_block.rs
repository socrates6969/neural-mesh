
// =============================================================================================
// REAL-text request payloads (appended by py/h10_mk_serve_ab.py)

struct Real {
    tasks: Vec<Vec<String>>,
    cluster: Vec<usize>,
    sents: Vec<String>,
}
static REAL: OnceLock<Real> = OnceLock::new();

fn real() -> &'static Real {
    REAL.get_or_init(|| {
        let mut tasks = Vec::new();
        let mut cluster = Vec::new();
        let mut sents = Vec::new();
        for tag in ["A", "B"] {
            let j: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_replay_{tag}.json")).expect("run py/h10_replay.py first")).unwrap();
            let ts = j["tasks"].as_array().unwrap();
            let cl = j["K"]["12"]["cluster"].as_array().unwrap();
            let step = (ts.len() / 300).max(1);
            for i in (0..ts.len()).step_by(step).take(300) {
                tasks.push(ts[i].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect());
                cluster.push(cl[i].as_u64().unwrap() as usize);
            }
            let c: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_corpus_{tag}.json")).unwrap()).unwrap();
            for d in c["docs"].as_array().unwrap().iter().step_by(37).take(120) {
                let t = d["text"].as_str().unwrap();
                if t.len() > 40 && t.len() < 400 {
                    sents.push(t.to_string());
                }
            }
        }
        Real { tasks, cluster, sents }
    })
}

fn real_features(r: &mut Rng, t: usize) -> String {
    let re = real();
    if r.below(6) == 0 {
        // a whole real doc paragraph as ONE feature string (quotes, dashes, unicode, punctuation)
        serde_json::to_string(&vec![re.sents[r.below(re.sents.len())].clone()]).unwrap()
    } else {
        let v = &re.tasks[t % re.tasks.len()];
        let k = 1 + r.below(v.len().min(4));
        serde_json::to_string(&v[..k]).unwrap()
    }
}

fn valid_real_request(r: &mut Rng, id: usize) -> String {
    let agents = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"];
    let ag = |r: &mut Rng| agents[r.below(agents.len())];
    let idpart = match r.below(4) {
        0 => String::new(),
        1 => format!(",\"id\":{id}"),
        2 => format!(",\"id\":\"req-{id}\""),
        _ => format!(",\"id\":{{\"n\":{id},\"x\":[1,2]}}"),
    };
    let t = r.below(real().tasks.len());
    match r.below(10) {
        0..=3 => {
            let mut extra = String::new();
            if r.below(4) == 0 {
                extra += &format!(",\"team\":{}", 1 + r.below(3));
            }
            if r.below(5) == 0 {
                extra += &format!(",\"seed\":{}", r.below(100000));
            }
            if r.below(6) == 0 {
                extra += &format!(",\"prior\":\"{}\"", ag(r));
            }
            if r.below(8) == 0 {
                extra += ",\"type\":\"sql\"";
            }
            if r.below(8) == 0 {
                extra += &format!(",\"candidates\":[\"{}\",\"{}\",\"{}\"]", ag(r), ag(r), ag(r));
            }
            if r.below(10) == 0 {
                extra += ",\"prior_k\":4";
            }
            format!("{{\"cmd\":\"route\",\"features\":{}{extra}{idpart}}}", real_features(r, t))
        }
        4..=6 => {
            let best = real().cluster[t] % 6;
            let a = if r.below(10) < 6 { best } else { r.below(7) };
            let rew = if a == best { 0.8 + 0.2 * r.f() } else { 0.3 * r.f() };
            let team = if r.below(3) == 0 { format!(",\"team\":[\"{}\",\"{}\"]", ag(r), ag(r)) } else { String::new() };
            format!("{{\"cmd\":\"feedback\",\"features\":{},\"agent\":\"a{a}\",\"reward\":{rew:.3}{team}{idpart}}}", real_features(r, t))
        }
        7 => format!("{{\"cmd\":\"stats\"{idpart}}}"),
        8 => {
            let c = match r.below(4) {
                0 => "\"eta\":0.02".to_string(),
                1 => format!("\"temperature\":{:.2}", 0.1 + r.f()),
                2 => format!("\"team\":{}", 1 + r.below(3)),
                _ => "\"decay\":0.001,\"epsilon\":0.05".to_string(),
            };
            format!("{{\"cmd\":\"config\",{c}{idpart}}}")
        }
        _ => format!("{{\"cmd\":\"flush\"{idpart}}}"),
    }
}

fn fuzz_real_line(r: &mut Rng, id: usize) -> Vec<u8> {
    let mut b = valid_real_request(r, id).into_bytes();
    for _ in 0..1 + r.below(3) {
        if b.is_empty() {
            break;
        }
        match r.below(5) {
            0 => {
                let k = r.below(b.len());
                b.truncate(k);
            }
            1 => {
                let k = r.below(b.len());
                b[k] ^= 1 << r.below(8);
            }
            2 => {
                let k = r.below(b.len() + 1);
                b.insert(k, (r.below(255) + 1) as u8);
            }
            3 => {
                let k = r.below(b.len());
                let l = 1 + r.below((b.len() - k).min(24));
                let seg: Vec<u8> = b[k..k + l].to_vec();
                for (j, c) in seg.into_iter().enumerate() {
                    b.insert(k + l + j, c);
                }
            }
            _ => {
                let k = r.below(b.len());
                b[k] = b[k].to_ascii_uppercase();
            }
        }
    }
    b.retain(|&c| c != b'\n');
    b.push(b'\n');
    b
}

/// The deterministic REAL-text request script (same shape as g6's `script`).
fn real_script(seed: u64, n_ops: usize, n_fuzz: usize, big: bool) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    let mut out: Vec<Vec<u8>> = Vec::new();
    out.push(line("{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\",\"a3\"],\"id\":\"warm\"}"));
    let errs = error_lines();
    let mut ei = 0;
    for i in 0..n_ops {
        out.push(line(&valid_real_request(&mut r, i)));
        if i % 7 == 3 && ei < errs.len() {
            out.push(errs[ei].clone());
            ei += 1;
        }
        if i % 97 == 50 {
            out.push(b"\n".to_vec());
        }
    }
    out.extend(errs[ei..].iter().cloned());
    for i in 0..n_fuzz {
        out.push(fuzz_real_line(&mut r, i));
    }
    out.push(line("{\"cmd\":\"stats\",\"id\":\"mid-stats\"}"));
    if big {
        out.extend(boundary_lines());
    }
    for i in 0..30 {
        out.push(line(&valid_real_request(&mut r, 90000 + i)));
    }
    out.push(line("{\"cmd\":\"stats\",\"id\":\"final-stats\"}"));
    out
}
