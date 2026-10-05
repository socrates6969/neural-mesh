//! h18_extract: spec-extraction accuracy on word problems (seen vs unseen phrasing), by level / op / domain, confidence
//! calibration and prose false positives. Usage: h18_extract --seeds 31-60 --tag held [--dump N]
//! SYNTHETIC: texts come from the h18 generator; the extractor lexicon is derived from the SEEN bank only.
mod h18_common;
use h18_common::world::*;
use neural_mesh::fin;
use h18_common::{arg, parse_range};
use rayon::prelude::*;
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Default, Clone)]
struct Cnt {
    n: usize,
    right: usize,
    abst: usize,
    wrong: usize,
    wrong_hi: usize,
    exact: usize,
}
impl Cnt {
    fn add(&mut self, o: &Cnt) {
        self.n += o.n;
        self.right += o.right;
        self.abst += o.abst;
        self.wrong += o.wrong;
        self.wrong_hi += o.wrong_hi;
        self.exact += o.exact;
    }
    fn s(&self) -> String {
        let n = self.n.max(1) as f64;
        format!("{:.3} / {:.3} / {:.3} ({:.3})", self.right as f64 / n, self.abst as f64 / n, self.wrong as f64 / n, self.wrong_hi as f64 / n)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seeds = parse_range(&args, "--seeds", (31, 60));
    let tag = arg(&args, "--tag", "held");
    let dump: usize = arg(&args, "--dump", "0").parse().unwrap_or(0);
    let bank_filter = arg(&args, "--bank", "any");
    let op_filter = arg(&args, "--op", "");
    let sp = arg(&args, "--spec", "");
    if !sp.is_empty() {
        println!("{:?}", fin::Args::parse(&sp).map(|a| fin::eval(0, &a).map(|o| o.render())));
        return;
    }
    let one = arg(&args, "--text", "");
    if !one.is_empty() {
        let e = h18_common::ext::extract(&one);
        println!("{:?}
{}", e, h18_common::ext::debug(&one));
        return;
    }
    let worlds = [WCfg::world_a(), WCfg::world_b()];
    let mut report = serde_json::Map::new();
    for cfg in &worlds {
        let ws: Vec<World> = seeds.par_iter().map(|s| gen_world(cfg, *s)).collect();
        let mut by: BTreeMap<String, Cnt> = BTreeMap::new();
        let mut calib = vec![(0usize, 0usize); 5];
        let mut prose = (0usize, 0usize, 0usize);
        let mut shown = 0;
        for w in &ws {
            for t in &w.tasks {
                if t.dom == 3 {
                    prose.0 += 1;
                    if t.ex.spec.is_some() {
                        prose.1 += 1;
                        if t.ex.conf >= 0.5 {
                            prose.2 += 1;
                        }
                    }
                    continue;
                }
                let mut c = Cnt { n: 1, ..Default::default() };
                match &t.ex.spec {
                    None => {
                        c.abst = 1;
                        if shown < dump && (bank_filter == "any" || (bank_filter == "seen") != t.unseen) && (op_filter.is_empty() || op_filter == t.op) {
                            shown += 1;
                            println!("--- ABSTAIN [{} L{} {}] {}\n  truth spec: {}\n  text: {}", t.op, t.level, if t.unseen { "UNSEEN" } else { "seen" }, cfg.name, t.spec, t.text);
                        }
                    }
                    Some(sp) => {
                        let ok = eval_spec(sp, 0).map(|v| is_correct(t, &v)).unwrap_or(false);
                        if sp == &t.spec {
                            c.exact = 1;
                        }
                        let cb = h18_common::ext::conf_bin(t.ex.conf);
                        calib[cb].0 += 1;
                        if ok {
                            c.right = 1;
                            calib[cb].1 += 1;
                        } else {
                            c.wrong = 1;
                            if t.ex.conf >= 0.6 {
                                c.wrong_hi = 1;
                            }
                            if shown < dump && (bank_filter == "any" || (bank_filter == "seen") != t.unseen) && (op_filter.is_empty() || op_filter == t.op) {
                                shown += 1;
                                println!("--- WRONG [{} L{} {}] {}\n  truth spec: {}\n  extracted : {} (conf {:.2})\n  text: {}", t.op, t.level, if t.unseen { "UNSEEN" } else { "seen" }, cfg.name, t.spec, sp, t.ex.conf, t.text);
                            }
                        }
                    }
                }
                let bank = if t.unseen { "unseen" } else { "seen" };
                for key in [
                    "all".to_string(),
                    format!("bank:{bank}"),
                    format!("L{}", t.level),
                    format!("L{}:{bank}", t.level),
                    format!("dom:{}", ["fin", "biz", "sci"][t.dom]),
                    format!("dom:{}:{bank}", ["fin", "biz", "sci"][t.dom]),
                    format!("op:{}", t.op),
                    format!("op:{}:{bank}", t.op),
                ] {
                    by.entry(key).or_default().add(&c);
                }
            }
        }
        println!("=== world {} ({} seeds {}-{}) : share right / abstain / wrong (wrong with conf>=.6) ===", cfg.name, seeds.len(), seeds[0], seeds[seeds.len() - 1]);
        for k in ["all", "bank:seen", "bank:unseen", "L0", "L1", "L2", "L3", "L4", "L0:unseen", "L1:unseen", "L2:unseen", "L3:unseen", "L4:unseen", "dom:fin", "dom:biz", "dom:sci", "dom:fin:unseen", "dom:biz:unseen", "dom:sci:unseen"] {
            if let Some(c) = by.get(k) {
                println!("  {:<16} n={:<6} {}   exact-string {:.3}", k, c.n, c.s(), c.exact as f64 / c.n.max(1) as f64);
            }
        }
        println!("  by op (seen | unseen right share):");
        let mut line = String::new();
        for o in h18_common::nl::OPS {
            let a = by.get(&format!("op:{}:seen", o.op)).cloned().unwrap_or_default();
            let b = by.get(&format!("op:{}:unseen", o.op)).cloned().unwrap_or_default();
            line.push_str(&format!("{}:{:.2}|{:.2}  ", o.op, a.right as f64 / a.n.max(1) as f64, b.right as f64 / b.n.max(1) as f64));
        }
        println!("    {}", line);
        let cal: Vec<String> = calib.iter().enumerate().map(|(i, (n, r))| format!("bin{}:{:.3}(n={})", i, *r as f64 / (*n).max(1) as f64, n)).collect();
        println!("  calibration P(right | conf bin <.3/.3-.5/.5-.7/.7-.9/>=.9): {}", cal.join(" "));
        println!("  prose tasks n={} extractor emitted a spec on {:.3} (conf>=.5 on {:.3})", prose.0, prose.1 as f64 / prose.0.max(1) as f64, prose.2 as f64 / prose.0.max(1) as f64);
        let jb: serde_json::Map<String, serde_json::Value> = by
            .iter()
            .map(|(k, c)| {
                (
                    k.clone(),
                    json!({"n": c.n, "right": c.right as f64 / c.n.max(1) as f64, "abstain": c.abst as f64 / c.n.max(1) as f64, "wrong": c.wrong as f64 / c.n.max(1) as f64, "wrong_hi": c.wrong_hi as f64 / c.n.max(1) as f64, "exact": c.exact as f64 / c.n.max(1) as f64}),
                )
            })
            .collect();
        report.insert(
            cfg.name.to_string(),
            json!({"by": jb, "prose_spec_rate": prose.1 as f64 / prose.0.max(1) as f64, "prose_conf_ge_half": prose.2 as f64 / prose.0.max(1) as f64, "calib": calib.iter().map(|(n, r)| json!([n, r])).collect::<Vec<_>>() }),
        );
    }
    let out = format!("results/h18_extract_{tag}.json");
    let _ = std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap_or_default());
    println!("wrote {out}");
}
