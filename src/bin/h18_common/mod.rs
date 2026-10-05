//! h18 shared pieces (R9-B): domain math tasks (finance / business / science / no-math prose), natural-language word
//! problems with seen / unseen phrasing banks, a rule-based spec extractor, the fin.rs tools + verifiers in the loop,
//! seeded worlds and the routing arms. All numbers are SYNTHETIC until validated on real traffic.
#![allow(dead_code)]
pub mod corpus;
pub mod ext;
pub mod nl;
pub mod world;

pub fn parse_range(args: &[String], key: &str, default: (u64, u64)) -> Vec<u64> {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key && i + 1 < args.len() {
            let v = &args[i + 1];
            if let Some((a, b)) = v.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.parse::<u64>(), b.parse::<u64>()) {
                    return (a..=b).collect();
                }
            }
            if let Ok(a) = v.parse::<u64>() {
                return vec![a];
            }
        }
        i += 1;
    }
    (default.0..=default.1).collect()
}
pub fn arg(args: &[String], key: &str, default: &str) -> String {
    let mut i = 0;
    while i < args.len() {
        if args[i] == key && i + 1 < args.len() {
            return args[i + 1].clone();
        }
        i += 1;
    }
    default.to_string()
}
