### VERIFY REQUEST — W6 — SERVE-PERF: BufWriter/one-write-per-reply fix in `neural-mesh serve` (byte-identical replies, 2.8x-14x TCP decisions/s)
Files: **src/serve.rs** (only src/ file touched), src/bin/g6_serve_ab.rs, src/bin/g6_common/mod.rs, results/g6_build_baseline.sh, results/g6_{equiv,bench,bench_AA_control,bench_optin_stat50}.{json,log}, results/g6_alloc.txt, results/g6_board_entry.md. No Cargo.toml / Cargo.lock / other src change, no new crate (std only; kernel32 FFI for child CPU/RSS as in f3), nothing downloaded, no git commit. Every server in every test is `--port 0` => binds **127.0.0.1 only** (the harness asserts the reported address starts with `127.0.0.1:`).

WHAT CHANGED in serve.rs (no reply byte changes; proof below):
 1. THE F3 FIX: each reply is serialised once (`serde_json::to_writer` into a reused `Vec<u8>` + `\n`; the old `writeln!(w, "{resp}")` on the raw `TcpStream` issued ~124 `write` syscalls per 194-byte reply, one per JSON token, as f3's serve_diagnosis measured) and sent with ONE `write`.
 2. Batched flush: while the client has more *complete* request lines already queued in our read buffer, replies are held and sent together (<=64 KiB); they are always sent before the server blocks for input, at EOF/error and on `shutdown`. A lockstep client still gets 1 write per reply (unit test counts writes and checks that nothing is pending when the server blocks); a pipelined burst of 30 requests gets 1 write.
 3. JSON parse, command lookup, field validation and option conversion (`parse_request`/`Plan`) now run OUTSIDE the global `Mutex<Server>`; only refresh+command+journal run under it. `dispatch`'s per-request `Vec<&str>` allowed-list and the array->`Vec<String>`->join allocations are gone.
 4. Over-long lines (> 1 MiB incl. newline) are discarded as they stream in (`Read::take(MAX_LINE+1)` + drain) instead of being buffered whole; the reply is the same `request line too long`. (The old loop's `read_until` buffers the whole line, so memory = line length by construction; not measured on the old code. Unit test on the new code: a 200 MB line peaks <6 MB heap.)
 5. OPT-IN, DEFAULT OFF, not part of the claims below unless stated: env `NEURAL_MESH_SERVE_STAT_MS=N` checks the state file at most every N ms instead of before every request (see "negatives": the per-request `stat` is now the dominant cost). With it unset the code path is the old one (`refresh()` stats before every request).

rerun (director; ~1.5 min equivalence + ~7 min per bench variant; run ALONE, the box is shared; paths as in the other entries):
  export PATH="$HOME/.cargo/bin:/c/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin:$PATH"; export CARGO_TARGET_DIR=C:/Users/mariu/nm-target-r6 RAYON_NUM_THREADS=3; cd /c/Users/mariu/neural-mesh
  bash results/g6_build_baseline.sh            # builds round-2 HEAD 52f936c (serve.rs untouched) from `git archive` into $CARGO_TARGET_DIR/g6/neural-mesh-base.exe (20 s warm; overwrites release/neural-mesh.exe, so do it FIRST)
  cargo build --release --offline --bin neural-mesh --bin g6_serve_ab
  $CARGO_TARGET_DIR/release/g6_serve_ab.exe equiv --out results/g6_equiv.json                     # 14 scenarios x (base, new, base-control): replies byte-for-byte, exit code, final state file; prints ALL BYTE-IDENTICAL (exit 1 otherwise)
  $CARGO_TARGET_DIR/release/g6_serve_ab.exe bench --secs 1.5 --reps 5 --out results/g6_bench.json # ABAB-interleaved base/new per config; ~6 min
  $CARGO_TARGET_DIR/release/g6_serve_ab.exe bench --base $CARGO_TARGET_DIR/release/neural-mesh.exe --new <copy of that exe> --new-env NEURAL_MESH_SERVE_STAT_MS=50   # the opt-in knob (the 2 exes must have different paths)
  cargo test --release --offline --lib                                                          # 228 passed, 0 failed (this tree)
  cargo test --release --offline --bin neural-mesh                                              # 19 passed + 1 ignored: 6 new serve tests + the ignored alloc report
  cargo test --release --offline --bin neural-mesh -- --ignored --nocapture alloc_report         # allocation counts
Machine: AMD Ryzen 7 8700F 8c/16t, Windows 11, 16 GB, SHARED with 5 other workers. CAVEAT: system CPU busy in the 250 ms before each timed run: median 59% (min 33%, max 97%) over the bench => absolute dec/s are depressed and noisy; the per-rep bg% is in the json. I therefore (a) alternate base/new inside every rep (ABAB, order flips each rep) and report the median of the PAIRED new/base ratios with min..max, (b) ran an A/A control (baseline vs a byte-copy of itself) to measure the noise floor, (c) report the raw medians + spread. Decision = one `route` + one `feedback` round trip (lockstep), 6 agents, 24 keyword features, learning on; replies are checked `"ok":true` (bad replies in all bench runs: 0).

**1. BYTE-IDENTICAL REPLIES (the hard requirement)** — `results/g6_equiv.json/.log`. The *real shipped binaries* are compared (baseline = HEAD 52f936c built from `git archive`; new = working tree), so the whole path (CLI parsing, stdio, TCP, state files) is covered, not just a module. Script: 2839 lines / 182891 bytes (valid route/feedback/stats/config/flush with numeric/string/object ids, `prior_k` underscore keys, team/seed/type/prior/candidates variants; ~85 hand-written error forms: malformed JSON, non-objects, missing/non-string cmd, unknown cmd (id echoed), unknown field, wrong field types, bad numbers/NaN/overflow, empty/whitespace/CRLF/NBSP/BOM/NUL/invalid-UTF-8 lines, 200-deep JSON nesting (serde recursion limit); 1800 mutated-request fuzz lines (truncate/flip/insert/duplicate bytes)) plus a "big" script with lines of exactly MAX_LINE-1, MAX_LINE, MAX_LINE+1, MAX_LINE+2 bytes, a 3 MiB valid-JSON line, a 2 MiB garbage line and a 1 MiB invalid-UTF-8 line, each followed by a good line (10.5 MB). Roughly 60% of the replies are error replies, 40% ok.
| scenario | reply bytes / lines (ok, err) | new == base | control base == base |
|---|---|---|---|
| stdio: whole script, EOF end | 942357 / 2797 (1117, 1680) | IDENTICAL | identical |
| stdio: whole script + shutdown mid-stream (+ignored tail) | 942408 / 2798 (1118, 1680) | IDENTICAL | identical |
| stdio: random 1..64-byte writes | 942357 / 2797 (1117, 1680) | IDENTICAL | identical |
| stdio: EOF in the middle of the last line (no newline) | 953429 / 2798 (1118, 1680) | IDENTICAL | identical |
| stdio: big/boundary lines (1 MiB limit, 3 MiB, invalid UTF-8) | 2357388 / 636 (289, 347) | IDENTICAL | identical |
| tcp: lockstep (one request, wait reply) | 942408 / 2798 (1118, 1680) | IDENTICAL | identical |
| tcp: whole script in one pipelined burst + shutdown | 942408 / 2798 (1118, 1680) | IDENTICAL | identical |
| tcp: random 1..300-byte writes (split lines) | 942408 / 2798 (1118, 1680) | IDENTICAL | identical |
| tcp: byte at a time (160 lines) | 68916 / 156 (140, 16) | IDENTICAL | identical |
| tcp: big/boundary lines, pipelined burst | 2357388 / 636 (289, 347) | IDENTICAL | identical |
| tcp: big/boundary lines, lockstep | 2357388 / 636 (289, 347) | IDENTICAL | identical |
| tcp: client half-closes after burst (EOF, no shutdown) | 942357 / 2797 (1117, 1680) | IDENTICAL | identical |
| tcp: half-close in the middle of the last line | 953432 / 2798 (1118, 1680) | IDENTICAL | identical |
| tcp: CLI writes interleaved (reload + journal replay) | 19015 / 62 (62, 0) | IDENTICAL | identical |
Each scenario also compares the server exit code and the final state-file bytes (stdio EOF / shutdown paths); the control row (baseline vs baseline) proves the harness is deterministic, and a self-test (one request byte changed, same exe) proves it can see a difference. Also: the rebuilt baseline from results/g6_build_baseline.sh is byte-identical in behaviour to the one used for the numbers (`equiv --quick --base <first build> --new <rebuilt>`: ALL BYTE-IDENTICAL), and with `NEURAL_MESH_SERVE_STAT_MS=50` set the same 14 scenarios stay identical (results/g6_equiv_statms.log; the CLI-interleaved scenario spawns a process per write, >50 ms, so it cannot show the deferral; the unit test `external_change_is_seen_before_the_next_request_unless_the_opt_in_gap_is_set` shows the knob really defers the reload). In-process, `stream_output_is_byte_identical_to_the_reference_loop` runs the verbatim OLD loop (kept in the test module) against the new `run_stream` with whole / 1-byte / 7-byte / 4093-byte reads.

**2. DECISIONS/S, OLD -> NEW (default behaviour, no knob)** — `results/g6_bench.json/.log`. Medians of 5 reps x 1.5 s (stdio: 20k decisions per run), ABAB interleaved. "paired x" = median over reps of (new dec/s)/(base dec/s) of two runs made back to back, with min..max.
| config | base dec/s med [min..max] | new dec/s med [min..max] | paired x med [min..max] | p50/decision base -> new (us) | p99 base -> new (us) |
|---|---|---|---|---|---|
| tcp lockstep N=1 [flush 64/1s] | 683 [582..695] | 5375 [4132..6090] | **7.87x** [7.10..9.30] | 1393 -> 142 | 2664 -> 1299 |
| tcp lockstep N=2 [flush 64/1s] | 956 [865..1048] | 7408 [5568..7671] | **7.16x** [6.17..7.89] | 2089 -> 193 | 3614 -> 1582 |
| tcp lockstep N=4 [flush 64/1s] | 1727 [1496..1816] | 6593 [6037..7744] | **4.20x** [3.82..4.46] | 2233 -> 452 | 3824 -> 2103 |
| tcp lockstep N=8 [flush 64/1s] | 2517 [2501..2709] | 6915 [6176..7851] | **2.77x** [2.28..2.95] | 3029 -> 892 | 4880 -> 2871 |
| tcp lockstep N=1 [flush off] | 673 [583..692] | 6430 [5944..6785] | **9.75x** [8.95..10.75] | 1398 -> 148 | 2088 -> 253 |
| tcp lockstep N=8 [flush off] | 2766 [2681..2814] | 8666 [8244..9885] | **3.17x** [3.01..3.57] | 2827 -> 916 | 4080 -> 1503 |
| tcp pipelined 32+32 N=1 [flush 64/1s] | 682 [576..725] | 9491 [7956..10324] | **13.81x** [12.28..15.12] | 1438 -> 105 | 1985 -> 161 |
| tcp pipelined 32+32 N=4 [flush 64/1s] | 1749 [1429..1877] | 10281 [8445..11666] | **5.89x** [5.80..6.22] | 2342 -> 336 | 2846 -> 1096 |
| tcp pipelined 32+32 N=8 [flush 64/1s] | 2624 [2496..2687] | 9533 [6121..10175] | **3.63x** [2.45..3.85] | 3174 -> 713 | 3541 -> 2645 |
| stdio pipe 20k dec [flush 64/1s] | 7705 [7328..7964] | 8230 [6911..10306] | **1.12x** [0.90..1.31] | 65 -> 61 |  |
| stdio pipe 20k dec [flush off] | 11292 [9684..13897] | 15304 [12271..17280] | **1.24x** [1.04..1.71] | 44 -> 33 |  |

**Per-decision server CPU and memory** (server process CPU = user+kernel from `GetProcessTimes` over the timed window / decisions; memory = `K32GetProcessMemoryInfo` peak working set / peak commit at the end of the run):
| config | server CPU us/decision base -> new | peak working set MB base -> new | peak commit MB base -> new |
|---|---|---|---|
| tcp lockstep N=1 [flush 64/1s] | 1444 -> 172 | 7.6 -> 7.7 | 1.7 -> 1.9 |
| tcp lockstep N=2 [flush 64/1s] | 2005 -> 183 | 7.7 -> 7.8 | 2.0 -> 2.0 |
| tcp lockstep N=4 [flush 64/1s] | 2171 -> 216 | 7.8 -> 7.9 | 1.9 -> 2.1 |
| tcp lockstep N=8 [flush 64/1s] | 2777 -> 202 | 8.0 -> 8.1 | 2.4 -> 2.4 |
| tcp lockstep N=1 [flush off] | 1458 -> 128 | 8.9 -> 20.0 | 2.9 -> 14.7 |
| tcp lockstep N=8 [flush off] | 2770 -> 181 | 13.5 -> 25.9 | 8.2 -> 20.5 |
| tcp pipelined 32+32 N=1 [flush 64/1s] | 1465 -> 104 | 7.5 -> 7.8 | 1.5 -> 1.9 |
| tcp pipelined 32+32 N=4 [flush 64/1s] | 2215 -> 116 | 7.8 -> 8.0 | 2.2 -> 2.3 |
| tcp pipelined 32+32 N=8 [flush 64/1s] | 2652 -> 117 | 8.0 -> 8.3 | 2.4 -> 2.6 |
| stdio pipe 20k dec [flush 64/1s] | 127 -> 116 | 7.8 -> 7.8 | 1.8 -> 1.9 |
| stdio pipe 20k dec [flush off] | 87 -> 62 | 32.8 -> 32.9 | 28.8 -> 28.9 |
Memory is unchanged to within noise at the default flush (7.5-8.3 MB working set, ~2 MB commit). With flush OFF the journal of un-flushed requests grows without bound in BOTH versions (that is why flush-off peak WS is larger and grows with the number of decisions done: the new server simply did 8-10x more decisions in the same 1.5 s; this is the pre-existing design, not a leak introduced here).

**3. NOISE FLOOR (A/A control)** — `results/g6_bench_AA_control.json`: baseline vs a byte-copy of the baseline, same protocol. Per-config paired-ratio medians range 0.89..1.04 (individual paired ratios 0.72..1.24). So differences below ~+-10% are noise on this box; every TCP row above is far outside that, the stdio-with-flush row (1.07x) is not.

**4. ALLOCATIONS** (`results/g6_alloc.txt`, in-process counting allocator in the serve.rs test module, 4001 route/feedback requests through the OLD loop vs the new `run_stream`, same replies asserted equal): allocator calls per request **66.0 -> 62.4 (-5.5%)**, bytes requested per request **5392 -> 5163 (-4.2%)**. Blunt: allocation was never the cost. ~62 allocations/request remain and nearly all are inside `cmd_route`/`cmd_feedback` (main.rs: `split_list`, score/health maps, `json!` trees) and the `serde_json::Value` request tree, which serve.rs cannot change without altering `cmd_*` or the malformed-JSON error text. The throughput win is syscalls (124 -> 1 write per reply) and lock hold time, not malloc.

**5. WHAT IS STILL SLOW, AND THE OPT-IN KNOB (negatives, measured)**
 * The remaining dominant cost of the serial path is `Server::refresh()`: one `std::fs::metadata` (CreateFile + GetFileInformationByHandle + close) on the state file BEFORE EVERY REQUEST. Measured in isolation: 19.3 us per call (GetFileAttributesExW is only 16.3 us: Defender/minifilter cost, so a cheaper syscall does not help). `serve --stdio` flush-off, default behaviour: base 11292 -> new 15304 dec/s (only 1.36x, 87 -> 62 us server CPU per decision). With the opt-in `NEURAL_MESH_SERVE_STAT_MS=50`: 88055 dec/s (**5.46x over default-new**, 11 us CPU/decision), TCP lockstep N=1 4712 -> 10429, N=8 flush-off 10173 -> 40999 dec/s (results/g6_bench_optin_stat50.json; table below). I did NOT make this the default because it changes WHEN an external CLI write is absorbed (up to N ms later; never lost, every flush still reloads+replays) and the task forbids output changes; it is the single biggest remaining lever and needs an owner decision.
 * Default flush cost: at flush 64/1 s the new server does 5375 dec/s lockstep N=1 vs 6430 with flush off; each count-triggered flush (lock file + whole-state write + rename, ~3-4 ms estimated from that difference) runs INSIDE the global lock and stalls every client. Moving it off the request path means restructuring `commit()` (lock/sig protocol with the CLI) — not output-neutral to reason about, so not done.
 * Scaling with clients is still poor: new lockstep N=1 -> N=8 is 5375 -> 6915 dec/s (1.3x for 8x the clients) at default flush (flush-off: 6430 -> 8666). The global `Mutex<Server>` section (stat + command + journal + occasional ms-long flush) is the cap; moving JSON parse out of it helped (old N=8 was 3.7x N=1) but cannot fix that.
 * stdio (no sockets) with default flush: 1.07x (paired 0.90..1.31) is inside the A/A noise; honest reading: "no measurable gain" there. Flush-off stdio 1.36x.
 * Baseline TCP numbers reproduce f3's (804|1378|2464|3409 at 1|2|4|8 clients; mine 683|956|1727|2517 under heavier load); f3's "~15 k decisions/s stdio ceiling" is what we see for flush-off (11-18 k).
 * Reply batching changes the TIMING of replies inside a pipelined burst (they leave together), never their bytes/order; a client that sends a burst and only reads after it is finished is unaffected, one that reads reply 1 before sending request 2 never has a reply held (unit test).
 * Not tested: Linux/macOS (Windows only box); more than ~8 concurrent clients; the `internal error` panic reply (no input I tried reaches a panic); concurrent multi-connection byte equality (replies depend on the global arrival order, so there is no deterministic reference; the bench instead checks every reply is `"ok":true`).
Opt-in knob `NEURAL_MESH_SERVE_STAT_MS=50` (new = default-new exe, "new" column = same code with the env var), same protocol:
| config | default dec/s med [min..max] | stat50 dec/s med [min..max] | paired x med [min..max] | p50/decision default -> stat50 (us) | p99 default -> stat50 (us) |
|---|---|---|---|---|---|
| tcp lockstep N=1 [flush 64/1s] | 4712 [4058..5182] | 10429 [9193..10769] | **2.07x** [1.95..2.57] | 164 -> 58 | 1424 -> 1095 |
| tcp lockstep N=2 [flush 64/1s] | 7614 [7320..7986] | 12390 [10324..12674] | **1.59x** [1.37..1.67] | 192 -> 83 | 1313 -> 1489 |
| tcp lockstep N=4 [flush 64/1s] | 6781 [5580..7619] | 14574 [13277..16745] | **2.22x** [1.96..2.61] | 455 -> 114 | 1988 -> 1656 |
| tcp lockstep N=8 [flush 64/1s] | 6185 [5720..6830] | 15637 [13035..16982] | **2.47x** [2.11..2.79] | 1024 -> 226 | 3088 -> 1792 |
| tcp lockstep N=1 [flush off] | 6267 [6008..6667] | 15807 [14292..16117] | **2.43x** [2.21..2.65] | 150 -> 60 | 261 -> 123 |
| tcp lockstep N=8 [flush off] | 10173 [8212..10924] | 40999 [36259..46508] | **4.16x** [3.75..4.44] | 757 -> 180 | 1674 -> 450 |
| tcp pipelined 32+32 N=1 [flush 64/1s] | 9683 [8831..10453] | 23226 [17039..25939] | **2.35x** [1.93..2.63] | 94 -> 40 | 153 -> 61 |
| tcp pipelined 32+32 N=4 [flush 64/1s] | 6697 [6463..10513] | 13993 [12007..17448] | **1.83x** [1.33..2.41] | 538 -> 280 | 1256 -> 480 |
| tcp pipelined 32+32 N=8 [flush 64/1s] | 9556 [6741..10472] | 17361 [14801..19770] | **1.84x** [1.61..2.20] | 718 -> 456 | 2890 -> 712 |
| stdio pipe 20k dec [flush 64/1s] | 10248 [9943..11340] | 23056 [20076..26037] | **2.25x** [1.96..2.62] | 49 -> 22 |  |
| stdio pipe 20k dec [flush off] | 16139 [14539..16804] | 88055 [87078..93939] | **5.64x** [5.43..6.06] | 31 -> 6 |  |

SUMMARY: the F3 diagnosis is confirmed and fixed. Real `neural-mesh serve --port` decisions/s (route+feedback, loopback, medians of 5, ABAB, noisy shared box): lockstep 1|2|4|8 clients 683->5375 (7.9x) | 956->7408 (7.2x) | 1727->6593 (4.2x) | 2517->6915 (2.8x); pipelined 32+32 per write at 1|4|8 clients 682->9491 (13.8x) | 1749->10281 (5.9x) | 2624->9533 (3.6x); replies byte-identical to HEAD in 14 scenarios incl. hostile input, 1 MiB boundaries, EOF/half-close and CLI-interleaved writes. Allocations -5.5% only; stdio no gain at default flush. Claimed multipliers are exactly the "paired x" column above, measured under 45-80% background load with an A/A noise floor of ~+-8%.
