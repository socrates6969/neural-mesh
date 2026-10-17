//! H15 workarounds that need NO crate change (all run against the experiment crate with FIX_MODE 0 = the current vendored behaviour,
//! except variant fix1 which is the proposed crate patch). One CSV row per build:
//!   variant,n,dim,seed,flag_first_two_upper,restarts_or_rebuilds,fail,total_ms
//!   h15_wk --wk none,restart2,conn,audit20,audit100,flat02,fix1 --n 500 --builds 5000
mod common;
use common::*;
use hnsw_exp::hnsw::{set_level_seed, Hnsw};
use std::time::Instant;

fn fresh(cfg: &Cfg, n: usize, lseed: u64) -> Hnsw<'static, f32, Cos> {
    set_level_seed(lseed);
    let mut h: Hnsw<'static, f32, Cos> = Hnsw::new(cfg.m, n.max(1), 16, cfg.efc, Cos);
    h.set_keeping_pruned(cfg.keep_pruned);
    h.set_extend_candidates(cfg.extend);
    if (cfg.level_scale - 1.0).abs() > 1e-12 {
        h.modify_level_scale(cfg.level_scale);
    }
    h
}

fn audit_ok(h: &Hnsw<'static, f32, Cos>, data: &[Vec<f32>], s: usize) -> bool {
    let n = data.len();
    let step = (n / s).max(1);
    (0..n).step_by(step).take(s).all(|i| h.search(&data[i], 1, 64).first().is_some_and(|x| x.d_id == i || x.distance <= 1e-6))
}

fn main() {
    let builds: u64 = arg("--builds", 1000);
    let n: usize = arg("--n", 500);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let mode = arg_s("--mode", "iid");
    let wks: Vec<String> = arg_s("--wk", "none").split(',').map(|s| s.to_string()).collect();
    let cfg0 = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: false, extend: false, level_scale: 1.0 };
    if !flag("--nohdr") {
        println!("variant,n,dim,seed,flag,extra,fail,total_ms");
    }
    for b in seed0..seed0 + builds {
        let data = gen_data(&mode, n, dim, b);
        let ls0 = level_seed_for(b);
        for wk in &wks {
            hnsw_exp::hnsw::FIX_MODE.store(if wk == "fix1" { 1 } else { 0 }, std::sync::atomic::Ordering::Relaxed);
            let cfg = Cfg { m: cfg0.m, efc: cfg0.efc, keep_pruned: false, extend: false, level_scale: if wk == "flat02" { 0.2 } else { 1.0 } };
            let flag0 = { let h = fresh(&cfg, n, ls0); h.insert_slice((data[0].as_slice(), 0)); h.insert_slice((data[1].as_slice(), 1)); h.get_point_indexation().get_layer_nb_point(0) == 0 };
            let t0 = Instant::now();
            let mut extra = 0u32;
            let build_plain = |ls: u64| {
                let h = fresh(&cfg, n, ls);
                for (i, v) in data.iter().enumerate() {
                    h.insert_slice((v.as_slice(), i));
                }
                h
            };
            let h = match wk.as_str() {
                "none" | "fix1" | "flat02" => build_plain(ls0),
                "restart2" => loop {
                    let ls = ls0.wrapping_add((extra as u64) * 0x5DEECE66D);
                    let h = fresh(&cfg, n, ls);
                    h.insert_slice((data[0].as_slice(), 0));
                    h.insert_slice((data[1].as_slice(), 1));
                    // public-API check: layer 0 holds only exact-level-0 points; if both first points are upper-level it is empty
                    if h.get_point_indexation().get_layer_nb_point(0) == 0 && extra < 20 {
                        extra += 1;
                        continue;
                    }
                    for (i, v) in data.iter().enumerate().skip(2) {
                        h.insert_slice((v.as_slice(), i));
                    }
                    break h;
                },
                "conn" => {
                    let mut h = build_plain(ls0);
                    while weak_components(&structure(&h, n)) > 1 && extra < 20 {
                        extra += 1;
                        h = build_plain(ls0.wrapping_add((extra as u64) * 0x5DEECE66D));
                    }
                    h
                }
                w if w.starts_with("audit") => {
                    let s: usize = w[5..].parse().unwrap();
                    let mut h = build_plain(ls0);
                    while !audit_ok(&h, &data, s) && extra < 20 {
                        extra += 1;
                        h = build_plain(ls0.wrapping_add((extra as u64) * 0x5DEECE66D));
                    }
                    h
                }
                _ => panic!("unknown wk {wk}"),
            };
            let total_ms = t0.elapsed().as_secs_f64() * 1e3;
            // verdict (outside the timed region): exact connectivity + 100-sample audit
            let st = structure(&h, n);
            let fl = flag0;
            let fail = weak_components(&st) > 1 || !audit_ok(&h, &data, 100);
            println!("{wk},{n},{dim},{b},{},{extra},{},{total_ms:.1}", fl as u8, fail as u8);
        }
    }
}
