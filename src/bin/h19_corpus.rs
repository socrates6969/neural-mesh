//! h19_corpus: print / dump deterministic synthetic corpora.  h19_corpus show --world paper|fin|biz --seed 1 --idx 0 --layout 2 --unseen
mod h19_common;
use h19_common::*;
use h19_common::gens::*;
use h19_common::genb::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("show");
    let world = match arg(&args, "--world").as_deref() { Some("paper") => World::Paper, Some("biz") => World::Biz, _ => World::Fin };
    let seed: u64 = arg(&args, "--seed").map(|s| s.parse().unwrap()).unwrap_or(1);
    let idx: u64 = arg(&args, "--idx").map(|s| s.parse().unwrap()).unwrap_or(0);
    let layout: u8 = arg(&args, "--layout").map(|s| s.parse().unwrap()).unwrap_or(1);
    let bank = if has(&args, "--unseen") { Bank::Unseen } else { Bank::Dev };
    match cmd {
        "show" => {
            let d = match world { World::Paper => gen_paper(seed, idx, bank, layout), World::Biz => gen_biz(seed, idx, bank, layout), _ => gen_fin(seed, idx, bank, layout) };
            println!("{}", d.text);
            println!("--- facts: {} claims: {} questions: {} distract: {}", d.facts.len(), d.claims.len(), d.qs.len(), d.n_distract);
            for f in d.facts.iter().take(40) {
                println!("{:?} {} {} {:?} [{}..{}] '{}'", f.kind, f.key, f.value, f.unit, f.lo, f.hi, if f.hi > f.lo && f.hi - f.lo < 40 { &d.text[f.lo..f.hi] } else { "" });
            }
            for q in &d.qs { println!("Q {:?} {} -> {}", q.op, q.text, q.truth); }
        }
        "hash" => {
            // determinism check: FNV-1a over the text of every generated doc
            let seeds = parse_seeds(&arg(&args, "--seeds").unwrap_or("1-8".into()));
            let mut h: u64 = 0xcbf29ce484222325;
            let mut n = 0usize;
            let mut chars = 0usize;
            let mut facts = 0usize;
            for &sd in &seeds {
                for w in [World::Paper, World::Fin, World::Biz] {
                    for l in 0..5u8 {
                        for b in [Bank::Dev, Bank::Unseen] {
                            for i in 0..3u64 {
                                let d = match w { World::Paper => gen_paper(sd, 1000 + i, b, l), World::Biz => gen_biz(sd, 1000 + i, b, l), _ => gen_fin(sd, 1000 + i, b, l) };
                                for c in d.text.bytes() { h ^= c as u64; h = h.wrapping_mul(0x100000001b3); }
                                n += 1;
                                chars += d.text.len();
                                facts += d.facts.len();
                            }
                        }
                    }
                }
            }
            println!("docs {} chars {} facts {} fnv1a {:016x}", n, chars, facts, h);
        }
        "dump" => {
            let mut out = String::new();
            for w in [World::Paper, World::Fin, World::Biz] {
                for l in [0u8, 2, 4] {
                    for b in [Bank::Dev, Bank::Unseen] {
                        let d = match w { World::Paper => gen_paper(1, 2000, b, l), World::Biz => gen_biz(1, 2000, b, l), _ => gen_fin(1, 2000, b, l) };
                        out.push_str(&format!("
######## SYNTHETIC world={} layout=L{} bank={:?} seed=1 idx=2000 facts={} questions={}
", w.name(), l, b, d.facts.len(), d.qs.len()));
                        out.push_str(&d.text);
                        for q in &d.qs { out.push_str(&format!("[Q {:?}] {} -> {}
", q.op, q.text, q.truth)); }
                    }
                }
            }
            std::fs::write("results/h19_corpus_samples.txt", out).unwrap();
            println!("wrote results/h19_corpus_samples.txt");
        }
        _ => {}
    }
}
